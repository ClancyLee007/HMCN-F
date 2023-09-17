import logging
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
import tensorflow as tf
logger = logging.getLogger()

class HMCN(tf.keras.Model):
    """
       implementation of HMCN-F, reference:http://proceedings.mlr.press/v80/wehrmann18a.html
    """

    def __init__(self,
                 label_total_num,
                 label_num_1,
                 label_num_2,
                 label_num_3,
                 feature_dim,
                 albert_emb_dim,
                 image_emb_dim,
                 mix_emb_dim,
                 batch_size,
                 LMF_rank,
                 fusion_dim,
                 beta=0.5,
                 dropout_rate=0.5,
                 *args,
                 **kwargs):
        super().__init__(*args, **kwargs)
        self.beta = beta
        self.fusion = tf.keras.layers.InputLayer(input_shape=fusion_dim, batch_size=batch_size)
        self.LMF = LMF(LMF_rank, batch_size, albert_emb_dim, image_emb_dim, mix_emb_dim, fusion_dim,
                       input_shape=[feature_dim])

        # layers
        with tf.name_scope("global"):
            self.global_dense_1 = tf.keras.Sequential(layers=[
                Dense(label_total_num, input_shape=[fusion_dim], activation=tf.nn.relu),
                Dropout(dropout_rate),
                BatchNormalization()],
                name='global_linear_with_dropout_and_bn_1'
            )
            self.global_dense_2 = tf.keras.Sequential(layers=[
                Dense(label_total_num, input_shape=[label_total_num + fusion_dim], activation=tf.nn.relu),
                Dropout(dropout_rate),
                BatchNormalization()],
                name='global_linear_with_dropout_and_bn_2'
            )
            self.global_dense_3 = tf.keras.Sequential(layers=[
                Dense(label_total_num, input_shape=[label_total_num + fusion_dim], activation=tf.nn.relu),
                Dropout(dropout_rate),
                BatchNormalization()],
                name='global_linear_with_dropout_and_bn_3'
            )
            self.global_sigmoid = tf.keras.layers.Dense(label_total_num, activation=tf.nn.sigmoid, name='output')

        with tf.name_scope("local"):
            self.local_dense_1 = linear_with_bn_and_dropout(label_num_1, activation=tf.nn.relu, name='dense_1')
            self.local_dense_2 = linear_with_bn_and_dropout(label_num_2, activation=tf.nn.relu, name='dense_2')
            self.local_dense_3 = linear_with_bn_and_dropout(label_num_3, activation=tf.nn.relu, name='dense_3')
            self.local_sigmoid_1 = tf.keras.layers.Dense(label_num_1, activation=tf.nn.sigmoid,
                                                         name='output_1')
            self.local_sigmoid_2 = tf.keras.layers.Dense(label_num_2, activation=tf.nn.sigmoid,
                                                         name='output_2')
            self.local_sigmoid_3 = tf.keras.layers.Dense(label_num_3, activation=tf.nn.sigmoid,
                                                         name='output_3')

        self.outputs = [self.global_sigmoid, self.local_sigmoid_1, self.local_sigmoid_2, self.local_sigmoid_3]

    def call(self, inputs, training=False):
        """
        :param inputs:video, image and text feature embedding. should be a dict.
        :param training:training model or not
        :return:output of sigmoid of global prediction
        """
        "multimodal fusion"
        fusion_feature = self.LMF(inputs)
        fusion_feature = self.fusion(fusion_feature)

        "global flow"
        self.A_G_1 = self.global_dense_1(fusion_feature)
        self.A_G_2 = self.global_dense_2(inputs=tf.concat([self.A_G_1, fusion_feature], axis=-1))
        self.A_G_3 = self.global_dense_3(inputs=tf.concat([self.A_G_2, fusion_feature], axis=-1))
        self.P_G = self.global_sigmoid(self.A_G_3)

        "first hierarchical flow"
        self.A_L_1 = self.local_dense_1(inputs=self.A_G_1)
        self.P_L_1 = self.local_sigmoid_1(self.A_L_1)

        "second hierarchical flow"
        self.A_L_2 = self.local_dense_2(inputs=self.A_G_2)
        self.P_L_2 = self.local_sigmoid_2(self.A_L_2)

        "third hierarchical flow"
        self.A_L_3 = self.local_dense_3(inputs=self.A_G_3)
        self.P_L_3 = self.local_sigmoid_3(self.A_L_3)

        "global probability"
        self.P_F = self.beta * tf.concat([self.P_L_1, self.P_L_2, self.P_L_3], axis=-1) + (1 - self.beta) * self.P_G
        return self.P_L_1, self.P_L_2, self.P_L_3, self.P_F

class linear_with_bn_and_dropout(tf.keras.layers.Layer):

    def __init__(self, unit, input_shape=None, activation=None, drop_rate=0.5, **kwargs):
        super(linear_with_bn_and_dropout, self).__init__(**kwargs)

        self.dense = tf.keras.layers.Dense(unit, input_shape=input_shape, activation=activation)
        self.dropout = tf.keras.layers.Dropout(drop_rate)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, inputs, training=False, **kwargs):
        dense_output = self.dense(inputs)
        if training:
            dropout_output = self.dropout(dense_output, training=training)
            bn_output = self.bn(dropout_output, training=training)
            output = bn_output
        else:
            output = dense_output

        return output


class LMF(tf.keras.layers.Layer):
    "implementation of Efficient Low-rank Multimodal Fusion. refer to https://arxiv.org/abs/1806.00064"

    def __init__(self, rank, batch_size, albert_emb_dim, image_emb_dim, mix_emb_dim, output_dim, **kwargs):
        super(LMF, self).__init__(**kwargs)

        self.batch_size = batch_size,
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

    def call(self, inputs):
        textual_emb = inputs[0]
        image_emb = inputs[1]
        video_emb = inputs[2]

        logger.info(f"albert emb tensor shape : {textual_emb.shape}")
        logger.info(f"image emb tensor shape : {image_emb.shape}")
        logger.info(f"video emb tensor shape : {video_emb.shape}")

        _vidio_h = tf.pad(video_emb, [[0, 0], [0, 1]])
        _textual_h = tf.pad(textual_emb, [[0, 0], [0, 1]])
        _image_h = tf.pad(image_emb, [[0, 0], [0, 1]])

        fusion_vidio = tf.matmul(_vidio_h, self.vidio_factor)
        fusion_textual = tf.matmul(_textual_h, self.textual_factor)
        fusion_image = tf.matmul(_image_h, self.image_factor)
        fusion_zy = fusion_vidio * fusion_textual * fusion_image

        output = tf.squeeze(tf.matmul(self.fusion_weights, tf.transpose(fusion_zy, perm=[1, 0, 2]))) + self.fusion_bias
        return output