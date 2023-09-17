import json
import os
import tensorflow as tf
import argparse
import pandas as pd
from model.model import HMCN
from model.HMCN import create_uncompiled_model
from meta.tag_tree import TagTree
from model.loss import Hierarchical_Violation_Loss
tf.config.list_physical_devices('GPU')  # 可使用的GPU列表
multiworker_strategy = tf.distribute.MultiWorkerMirroredStrategy()  # 设置分布式训练策略


parser = argparse.ArgumentParser()
parser.add_argument("--train", type=bool, help="whether training model or not")
parser.add_argument("--evaluate", type=bool, help="whether only evaluate model or not")
parser.add_argument("--predict", type=bool, help="whether only predict or not")
parser.add_argument("--train_data", type=str, help="data folder path for training")
parser.add_argument("--test_data", type=str, help="data folder for evaluation")
parser.add_argument("--predict_data", type=str, help="data folder for prediction")
parser.add_argument("--batch_size", type=int, help="batch size")
parser.add_argument("--epoch", type=int, help="number of training epoch")
parser.add_argument("--model_config", type=str, help="model creation config")
parser.add_argument("--model_dir", type=str, help="dir for output model")
parser.add_argument("--output_dir", type=str, help="dir for history excel")
parser.add_argument("--tag_tree_file", type=str, help="tag tree file",
                    default="D://project/hmcn/meta/tag_content_tree_map.txt")
parser.add_argument("--learning_rate", type=float, help="learning rate")
parser.add_argument("--cpu_num", type=int, help="number of cpu will be used", default=12)
FLAGS, unparsed = parser.parse_known_args()
logger = tf.get_logger()
tagTree = TagTree(FLAGS.tag_tree_file)
tf.config.list_physical_devices('GPU')


def tfrecord_parse(tfrecord):
    feature_schema = {
        "albert_emb": tf.io.FixedLenFeature(1024, dtype=tf.float32),
        "image_emb": tf.io.FixedLenFeature(2048, dtype=tf.float32),
        "mix_emb": tf.io.FixedLenFeature(1024, dtype=tf.float32),
        "tags": tf.io.VarLenFeature(dtype=tf.int64)
    }

    parsed_data = tf.io.parse_single_example(tfrecord, feature_schema)
    ori_tags = tf.sparse.to_dense(parsed_data.pop("tags"))
    features = parsed_data

    ##去除不在标签体系中的标签
    tags = tf.gather_nd(ori_tags, tf.where(ori_tags > 0))
    label_G = tf.reduce_sum(tf.one_hot(indices=tags, depth=tagTree.total_tag_num, axis=-1), axis=0)
    label_1 = label_G[:tagTree.tagL1_num]
    label_2 = label_G[tagTree.tagL1_num: tagTree.tagL1_num + tagTree.tagL2_num]
    label_3 = label_G[tagTree.tagL1_num + tagTree.tagL2_num:]

    labels = {
        "global_prob": label_G,
        "local_prob_1": label_1,
        "local_prob_2": label_2,
        "local_prob_3": label_3
    }

    return [features, labels]


def train_input_fn(epoch):
    data_dir = FLAGS.train_data
    # file_list = os.listdir(data_dir)
    dataset = tf.data.TFRecordDataset(data_dir, num_parallel_reads=FLAGS.cpu_num)

    logger.info("dataset.repeat(%d)" % (epoch))
    dataset = dataset.repeat(epoch).shuffle(buffer_size=FLAGS.batch_size * 10, reshuffle_each_iteration=True)
    dataset = dataset.map(tfrecord_parse).batch(FLAGS.batch_size, drop_remainder=True).prefetch(1)

    return dataset


def eval_input_fn():
    data_dir = FLAGS.test_data
    # file_list = os.listdir(data_dir)
    dataset = tf.data.TFRecordDataset(data_dir, num_parallel_reads=FLAGS.cpu_num)
    dataset = dataset.map(tfrecord_parse).batch(FLAGS.batch_size, drop_remainder=True).prefetch(1)

    return dataset


def create_HMCN(config):
    with open(config, "r") as f:
        config_dict = json.load(f)
    feature_dim = config_dict["feature_dim"]
    label_num_1 = config_dict["label_num_1"]
    label_num_2 = config_dict["label_num_2"]
    label_num_3 = config_dict["label_num_3"]
    label_total_num = config_dict["label_total_num"]
    LMF_rank = config_dict["LMF_rank"]
    fusion_dim = config_dict["LMF_output_dim"]
    feature_schema = config_dict["feather_schema"]
    batch_size = FLAGS.batch_size
    feature_info = {}
    for info in feature_schema:
        feature_info[info["name"]] = int(info["dim"])

    # hmcn = HMCN(label_total_num, label_num_1, label_num_2, label_num_3, feature_dim, feature_info["albert_emb"],
    #             feature_info["image_emb"], feature_info["mix_emb"],
    #             LMF_rank, batch_size, fusion_dim)
    hmcn = create_uncompiled_model(batch_size, feature_info["albert_emb"], feature_info["image_emb"],
                                   feature_info["mix_emb"], LMF_rank, fusion_dim, label_total_num, label_num_1,
                                   label_num_2, label_num_3)
    hmcn.summary()

    return hmcn


def train_model():
    # tf.debugging.experimental.enable_dump_debug_info(
    #     "/tmp/tfdbg2_logdir",
    #     tensor_debug_mode="FULL_HEALTH",
    #     circular_buffer_size=-1)
    with multiworker_strategy.scope():
        model_config = FLAGS.model_config
        hmcn = create_HMCN(model_config)

        optimizer = tf.keras.optimizers.Adam(learning_rate=FLAGS.learning_rate)
        hmcn.compile(
            optimizer=optimizer,
            loss={
                "global_prob": [tf.keras.losses.BinaryCrossentropy(),
                                Hierarchical_Violation_Loss()],
                "local_prob_1": tf.keras.losses.BinaryCrossentropy(),
                "local_prob_2": tf.keras.losses.BinaryCrossentropy(),
                "local_prob_3": tf.keras.losses.BinaryCrossentropy(),
            },
            metrics={
                "local_prob_1": [
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.CategoricalAccuracy()
                ],
                "local_prob_2": [
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall(),
                    tf.keras.metrics.CategoricalAccuracy()
                ],
                "global_prob": [
                    tf.keras.metrics.Precision(),
                    tf.keras.metrics.Recall()
                ]
            }
        )
        tf.keras.utils.plot_model(hmcn, r"C:\Users\likanglu\Desktop\hmcn.png")

        train_data = train_input_fn(FLAGS.epoch)
        eval_data = eval_input_fn()

        ##save the best model
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=FLAGS.model_dir,
            save_best_only=True,
            monitor='val_loss',
            mode='min'
        )

        history = hmcn.fit(train_data,
                           validation_data=eval_data,
                           epochs=FLAGS.epoch,
                           batch_size=FLAGS.batch_size,
                           callbacks=[model_checkpoint_callback],
                           workers=FLAGS.cpu_num,
                           use_multiprocessing=True)

    with tf.device("/device:CPU:0"):
        df = pd.DataFrame(history.history)
        historyPath = os.path.join(FLAGS.output_dir, 'history.xlsx')
        df.to_excel(historyPath, index=False)
    # hmcn.evaluate(eval_data)


if __name__ == "__main__":
    train_model()
