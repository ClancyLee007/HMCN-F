import logging
import os
from collections import defaultdict
from varname import nameof
from meta.tag_tree import TagTree
import tensorflow as tf
logging.getLogger().setLevel(logging.INFO)
def process_tags(tags):
    for idx, tag in enumerate(tags):
        if tag == "古装剧":
            tags[idx] = "古偶剧"
def encode_tag_label(file, info_dict):
    for line in file:
        raw = line.strip("\r\n").split("\t")
        if len(raw) != 3:
            logging.warning(f"{line} format incorrect,skip..")
            continue
        qipuid, text, tag = raw[0], raw[1], raw[2]
        if qipuid == 0: continue

        tags = tag.split("\3")
        process_tags(tags)
        taglist = list(map(lambda x: tagTree.tag_id_map.get(x, 0), tags))
        info_dict[qipuid].update({"tags": taglist})
    logging.info(f"{nameof(info_dict)},带标签视频{len(info_dict)}个")

    return info_dict


def decode_emb(dir, train_dict, test_dict):
    file_list = os.listdir(dir)
    for file in file_list:
        with open(os.path.join(dir, file)) as f:
            for line in f:
                embinfo = line.strip("\r\n").split("\t")
                if len(embinfo) == 3:
                    qipuid, emb_type, emb = tuple(embinfo)
                else:
                    logging.warning(f"{line} format incorrect,skip..")
                    continue

                if qipuid == 0: continue

                if qipuid in train_dict.keys():
                    logging.debug(f"{qipuid}'s {emb_type} decode successfully, writen in train set.\n{train_dict[qipuid]}")
                    train_dict[qipuid].update({emb_type: list(map(float, emb.split(",")))})
                elif qipuid in test_dict.keys():
                    logging.debug(f"{qipuid}'s {emb_type} decode successfully, writen in test set.\n{test_dict[qipuid]}")
                    test_dict[qipuid].update({emb_type: list(map(float, emb.split(",")))})
                else:
                    logging.debug(f"{qipuid} decode fail")
            f.close()


def filter(video_info):
    """
    对信息不全的视频进行清洗
    :return: dict
    """
    def is_all_none(lt):
        for item in lt:
            if item is not None:
                return False
        return True
    del_set = set()
    for qipuid, info in video_info.items():
        flag = 0
        if "image_embedding" not in info.keys():
            # flag = 1
            info["image_embedding"] = [0.0] * 2048
        if "albert_embedding" not in info.keys():
            # flag = 1
            info["albert_embedding"] = [0.0] * 1024
        if "mix_embedding" not in info.keys():
            # flag = 1
            info["mix_embedding"] = [0.0] * 1024
        if "tags" not in info.keys():
            flag = 1
        elif is_all_none(info["tags"]):
            flag = 1
        if flag:
            del_set.add(qipuid)
    logging.info(f"原有视频{len(video_info)}个")

    for qipuid in del_set:
        del video_info[qipuid]
    logging.info(f"信息完整视频{len(video_info)}个")

def write_tfrecord(path, video_info):
    writer = tf.io.TFRecordWriter(path=path)
    for qipuid in video_info.keys():
        info = video_info[qipuid]
        tfrecord = tf.train.Example(features=tf.train.Features(
            feature={
                "tags": tf.train.Feature(int64_list=tf.train.Int64List(value=info["tags"])),
                "image_emb": tf.train.Feature(float_list=tf.train.FloatList(value=info["image_embedding"])),
                "albert_emb": tf.train.Feature(float_list=tf.train.FloatList(value=info["albert_embedding"])),
                "mix_emb": tf.train.Feature(float_list=tf.train.FloatList(value=info["mix_embedding"])),
            }
        )).SerializeToString()
        writer.write(tfrecord)
    writer.close()


if __name__ == "__main__":
    train_label_file = "../data/trainDataSegmentedChar.txt.train"
    test_label_file = "../data/trainDataSegmentedChar.txt.test"
    image_emb_dir = "../data/image_emb"
    albert_emb_dir = "../data/albert_emb"
    mix_emb_dir = "../data/mix_emb"
    train_tfrecord_path = "../data/train.tfrecords"
    test_tfrecord_path = "../data/test.tfrecords"
    tagTree = TagTree()
    train_video_info = defaultdict(dict)
    test_video_info = defaultdict(dict)

    train_label = open(train_label_file, encoding="utf8")
    test_label = open(test_label_file, encoding="utf8")
    ##写入每个视频label
    encode_tag_label(train_label, train_video_info)
    encode_tag_label(test_label, test_video_info)
    ##写入albert,image,mix的embedding
    decode_emb(image_emb_dir, train_video_info, test_video_info)
    decode_emb(albert_emb_dir, train_video_info, test_video_info)
    decode_emb(mix_emb_dir, train_video_info, test_video_info)
    ##去掉信息不全的样本
    filter(train_video_info)
    filter(test_video_info)

    write_tfrecord(train_tfrecord_path, train_video_info)
    write_tfrecord(test_tfrecord_path, test_video_info)


