from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense, Add, Dropout, BatchNormalization
import tensorflow as tf
import logging

logger = logging.getLogger()


def create_uncompiled_model(batch_size, albert_dim, image_dim, mix_dim, LMF_rank, fusion_dim, label_total_num,
                            label_num_1, label_num_2, label_num_3, beta=0.5):
    albert_emb = Input(shape=[albert_dim], batch_size=batch_size, name="albert_emb")
    image_emb = Input(shape=[image_dim], batch_size=batch_size, name="image_emb")
    mix_emb = Input(shape=[mix_dim], batch_size=batch_size, name="mix_emb")

    fusion_feature = LMF(LMF_rank, batch_size, albert_dim, image_dim, mix_dim, fusion_dim,
                         name="multimodal_fusion")([albert_emb, image_emb, mix_emb])

    "global flow"
    A_G_1 = linear_with_bn_and_dropout(label_total_num, input_shape=[fusion_dim], activation=tf.nn.relu,
                                       name="global_dense_1")(fusion_feature)
    A_G_2 = linear_with_bn_and_dropout(label_total_num, input_shape=[label_total_num + fusion_dim],
                                       activation=tf.nn.relu,
                                       name="global_dense_2")(tf.concat([A_G_1, fusion_feature], axis=-1))
    A_G_3 = linear_with_bn_and_dropout(label_total_num, input_shape=[label_total_num + fusion_dim],
                                       activation=tf.nn.relu,
                                       name="global_dense_3")(tf.concat([A_G_2, fusion_feature], axis=-1))
    P_G = Dense(label_total_num, input_shape=[label_total_num], activation=tf.nn.sigmoid, name='logit_G')(A_G_3)

    "first hierarchical flow"
    A_L_1 = linear_with_bn_and_dropout(label_num_1, input_shape=[label_total_num], activation=tf.nn.relu,
                                       name="local_dense_1")(A_G_1)
    P_L_1 = Dense(label_num_1, input_shape=[label_num_1], activation=tf.nn.sigmoid, name="local_prob_1")(A_L_1)

    "second hierarchical flow"
    A_L_2 = linear_with_bn_and_dropout(label_num_2, input_shape=[label_total_num], activation=tf.nn.relu,
                                       name="local_dense_2")(A_G_2)
    P_L_2 = Dense(label_num_2, input_shape=[label_num_2], activation=tf.nn.sigmoid, name="local_prob_2")(A_L_2)

    "third hierarchical flow"
    A_L_3 = linear_with_bn_and_dropout(label_num_3, input_shape=[label_total_num], activation=tf.nn.relu,
                                       name="local_dense_3")(A_G_3)
    P_L_3 = Dense(label_num_3, input_shape=[label_num_3], activation=tf.nn.sigmoid, name="local_prob_3")(A_L_3)

    "global probalility"
    # P_F = HMCN_output(beta=beta, name="global_prob")([P_L_1, P_L_2, P_L_3, P_G])
    P_F = Add(name="global_prob")[beta * (tf.concat([P_L_1, P_L_2, P_L_3], axis=-1)), (1 - beta) * P_G]

    hmcn = Model(
        inputs=[albert_emb, image_emb, mix_emb],
        outputs=[P_L_1, P_L_2, P_L_3, P_F],
        name="tag_tree_classifier"
    )
    # tf.keras.utils.plot_model(hmcn, r"C:\Users\likanglu\Desktop\hmcn.png")

    return hmcn


class LMF(tf.keras.layers.Layer):
    """
    implementation of Efficient Low-rank Multimodal Fusion. refer to https://arxiv.org/abs/1806.00064
    """

    def __init__(self, rank, batch_size, albert_emb_dim, image_emb_dim, mix_emb_dim, output_dim, **kwargs):
        super(LMF, self).__init__(**kwargs)

        self.batch_size = batch_size
        self.rank = rank
        self.output_dim = output_dim

        initializer = tf.random_normal_initializer(mean=0, stddev=1)
        self.vidio_factor = tf.Variable(
            initial_value=initializer(shape=[self.rank, mix_emb_dim + 1, self.output_dim], dtype=tf.float32),
            trainable=True)
        self.textual_factor = tf.Variable(
            initial_value=initializer(shape=[self.rank, albert_emb_dim + 1, self.output_dim], dtype=tf.float32),
            trainable=True)
        self.image_factor = tf.Variable(
            initial_value=initializer(shape=[self.rank, image_emb_dim + 1, self.output_dim], dtype=tf.float32),
            trainable=True)
        self.fusion_weights = tf.Variable(initial_value=initializer(shape=[1, self.rank], dtype=tf.float32),
                                          trainable=True)
        self.fusion_bias = tf.Variable(initial_value=initializer(shape=[1, self.output_dim], dtype=tf.float32),
                                       trainable=True)

    def call(self, inputs, **kwargs):
        textual_emb = inputs[0]
        image_emb = inputs[1]
        video_emb = inputs[2]

        logger.info(f"albert emb tensor shape : {textual_emb.shape}")
        logger.info(f"image emb tensor shape : {image_emb.shape}")
        logger.info(f"video emb tensor shape : {video_emb.shape}")

        _vidio_h = tf.pad(video_emb, [[0, 0], [0, 1]], constant_values=1)
        _textual_h = tf.pad(textual_emb, [[0, 0], [0, 1]], constant_values=1)
        _image_h = tf.pad(image_emb, [[0, 0], [0, 1]], constant_values=1)

        fusion_vidio = tf.matmul(_vidio_h, self.vidio_factor)
        fusion_textual = tf.matmul(_textual_h, self.textual_factor)
        fusion_image = tf.matmul(_image_h, self.image_factor)
        fusion_zy = fusion_vidio * fusion_textual * fusion_image

        output = tf.squeeze(tf.matmul(self.fusion_weights, tf.transpose(fusion_zy, perm=[1, 0, 2]))) + self.fusion_bias
        return output


class linear_with_bn_and_dropout(tf.keras.layers.Layer):

    def __init__(self, unit, input_shape=None, activation=None, drop_rate=0.5, **kwargs):
        super(linear_with_bn_and_dropout, self).__init__(**kwargs)

        self.dense = Dense(unit, input_shape=input_shape, activation=activation)
        self.dropout = Dropout(drop_rate)
        self.bn = BatchNormalization()

    def call(self, inputs, training=False, **kwargs):
        dense_output = self.dense(inputs)
        if training:
            dropout_output = self.dropout(dense_output, training=training)
            bn_output = self.bn(dropout_output, training=training)
            output = bn_output
        else:
            output = dense_output

        return output


class HMCN_output(tf.keras.layers.Layer):
    def __init__(self, beta=0.5, **kwargs):
        super(HMCN_output, self).__init__(**kwargs)
        self.beta = beta

    def call(self, inputs, **kwargs):
        local_1 = inputs[0]
        local_2 = inputs[1]
        local_3 = inputs[2]
        glob = inputs[3]

        return tf.add(self.beta * (tf.concat([local_1, local_2, local_3], axis=-1)), (1 - self.beta) * glob)
