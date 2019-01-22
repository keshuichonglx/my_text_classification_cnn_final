#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    :   ModelCnn.py
@Time    :   2018/6/10 12:23
@Software:   PyCharm
@Author  :   Li Xiang
@Desc    :   卷积神经网络模型
"""
import tensorflow as tf
from tensorflow.contrib import layers

__time__ = '2018/6/10 12:23'

class ModleCnn_3(object):
    """
    本方案仅仅考虑w2v模型输入参数进行设计
    """

    def __init__(self, config):
        self.config = config

        # 输入数据占位符-
        self.input_voc_x = tf.placeholder(tf.int32, [None, self.config.voc_seq_length], name='input_voc_x')
        self.input_w2v_x = tf.placeholder(tf.int32, [None, self.config.w2v_seq_length], name='input_w2v_x')
        self.input_cix_x = tf.placeholder(tf.int32, [None, self.config.cix_seq_length], name='input_cix_x')
        self.input_f2_x = tf.placeholder(tf.float32, [None, self.config.f2_dimension], name='input_f2_x')

        self.input_y = tf.placeholder(tf.int32, [None, self.config.classes_num], name='input_y')
        self.embedding_w2v = tf.placeholder(tf.float32, [self.config.w2v_vocab_size, self.config.w2v_dimension], name='embedding_w2v')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.learning_rate = tf.placeholder(tf.float32, name='learning_rate')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        #with tf.device('/cpu:0'):
        with tf.device('/gpu:0'):
            embedding_voc = tf.get_variable('embedding_voc', [self.config.voc_vocab_size, self.config.voc_dimension])
            embedding_cix = tf.get_variable('embedding_cix', [self.config.cix_vocab_size, self.config.cix_dimension])
            embedding_voc_inputs = tf.nn.embedding_lookup(embedding_voc, self.input_voc_x)
            embedding_cix_inputs = tf.nn.embedding_lookup(embedding_cix, self.input_cix_x)
            # lookup生成张量
            embedding_w2v_inputs = tf.nn.embedding_lookup(self.embedding_w2v, self.input_w2v_x)

        with tf.name_scope("cnn_voc"):
            # 卷积1
            conv_voc1 = tf.layers.conv1d(embedding_voc_inputs, self.config.filters_num_voc1, self.config.kernel_size_voc1, name='conv11')
            # 池化
            conv_voc1 = tf.layers.max_pooling1d(inputs=conv_voc1, pool_size=self.config.pool_size_voc, strides=4, padding='same')
            # 卷积2
            conv_voc2 = tf.layers.conv1d(conv_voc1, self.config.filters_num_voc2, self.config.kernel_size_voc2, name='conv12')
            # 全局最大池化
            conv_voc2 = tf.reduce_max(conv_voc2, reduction_indices=[1], name='same12')

            # cnn的全连接
            fc_voc = tf.layers.dense(conv_voc2, self.config.hidden_neuron_num_voc, name='conv_fc1')
            fc_voc = tf.contrib.layers.dropout(fc_voc, self.keep_prob)
            fc_voc = tf.nn.relu(fc_voc)
            #fc_voc = tf.nn.leaky_relu(fc_voc)

        with tf.name_scope("cnn_w2v"):
            # 卷积1
            conv_w2v1 = tf.layers.conv1d(embedding_w2v_inputs, self.config.filters_num_w2v1, self.config.kernel_size_w2v1, name='conv21')
            # 池化
            conv_w2v1 = tf.layers.max_pooling1d(inputs=conv_w2v1, pool_size=self.config.pool_size_w2v, strides=4, padding='same')
            # 卷积2
            conv_w2v2 = tf.layers.conv1d(conv_w2v1, self.config.filters_num_w2v2, self.config.kernel_size_w2v2, name='conv22')
            # 全局最大池化
            conv_w2v2 = tf.reduce_max(conv_w2v2, reduction_indices=[1], name='same22')

            # cnn的全连接
            fc_w2v = tf.layers.dense(conv_w2v2, self.config.hidden_neuron_num_w2v, name='conv_fc2')
            fc_w2v = tf.contrib.layers.dropout(fc_w2v, self.keep_prob)
            fc_w2v = tf.nn.relu(fc_w2v)
            #fc_w2v = tf.nn.leaky_relu(fc_w2v)

        with tf.name_scope("cnn_cix"):
            # 卷积1
            conv_cix1 = tf.layers.conv1d(embedding_cix_inputs, self.config.filters_num_cix1, self.config.kernel_size_cix1, name='conv31')
            # 池化
            conv_cix1 = tf.layers.max_pooling1d(inputs=conv_cix1, pool_size=self.config.pool_size_cix, strides=4, padding='same')
            # 卷积2
            conv_cix2 = tf.layers.conv1d(conv_cix1, self.config.filters_num_cix2, self.config.kernel_size_cix2, name='conv32')
            # 全局最大池化
            conv_cix2 = tf.reduce_max(conv_cix2, reduction_indices=[1], name='same32')

            # cnn的全连接
            fc_cix = tf.layers.dense(conv_cix2, self.config.hidden_neuron_num_cix, name='conv_fc3')
            fc_cix = tf.contrib.layers.dropout(fc_cix, self.keep_prob)
            fc_cix = tf.nn.relu(fc_cix)
            #fc_cix = tf.nn.leaky_relu(fc_cix)

        with tf.name_scope("fnn_f2"):
            fc_f2 = tf.layers.dense(self.input_f2_x, self.config.hidden_neuron_num_f2, name='f2_fc3')
            fc_f2 = tf.contrib.layers.dropout(fc_f2, self.keep_prob)
            fc_f2 = tf.nn.relu(fc_f2)
            #fc_f2 = tf.nn.leaky_relu(fc_f2)

        with tf.name_scope("fnn_fin"):
            # 拼接所有数据
            concat1 = tf.concat([fc_voc, fc_w2v], 1, name='concat1')
            concat2 = tf.concat([concat1, fc_cix], 1, name='concat1')
            concat = tf.concat([concat2, fc_f2], 1, name='concat')

            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(concat, self.config.hidden_neuron_num_fin_1, name='ff_fc4')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)
            #fc = tf.nn.leaky_relu(fc)

            if self.config.hidden_neuron_num_fin_2 > 0:
                fc = tf.layers.dense(fc, self.config.hidden_neuron_num_fin_2, name='ff_fc5')
                fc = tf.contrib.layers.dropout(fc, self.keep_prob)
                fc = tf.nn.relu(fc)
                #fc = tf.nn.leaky_relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.classes_num, name='fc_f')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

class ModleCnn_2(object):
    """
    本方案仅仅考虑w2v模型输入参数进行设计
    """

    def __init__(self, config):
        self.config = config

        # 输入数据占位符
        self.input_x = tf.placeholder(tf.int32, [None, self.config.w2v_seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, self.config.classes_num], name='input_y')
        #self.input_x2 = tf.placeholder(tf.float32, [None, self.config.tfidf_dimension], name='input_x2')
        self.input_x2 = tf.placeholder(tf.float32, [None, self.config.f2_dimension], name='input_x2')
        self.embedding = tf.placeholder(tf.float32, [self.config.w2v_vocab_size, self.config.w2v_dimension],name='embedding')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        #with tf.device('/cpu:0'):
        with tf.device('/gpu:0'):
            # lookup生成张量
            embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.filters_num, self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
            # 尝试拼接一个试试看
            concat = tf.concat([gmp, self.input_x2], 1, name='concat')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(concat, self.config.hidden1_neuron_num, name='fc1')
            #fc = tf.layers.dense(gmp, self.config.hidden1_neuron_num, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            if self.config.hidden2_neuron_num > 0:
                # 全连接层fc2
                fc = tf.layers.dense(fc, self.config.hidden2_neuron_num, name='fc2')
                fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.classes_num, name='fc_f')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

class ModleCnn_1(object):
    """
    本方案仅仅考虑w2v模型输入参数进行设计
    """

    def __init__(self, config):
        self.config = config

        # 输入数据占位符
        self.input_x = tf.placeholder(tf.int32, [None, self.config.w2v_seq_length], name='input_x')
        self.input_y = tf.placeholder(tf.int32, [None, self.config.classes_num], name='input_y')
        self.embedding = tf.placeholder(tf.float32, [self.config.w2v_vocab_size, self.config.w2v_dimension],name='embedding')
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')

        self.cnn()

    def cnn(self):
        """CNN模型"""
        with tf.device('/cpu:0'):
        #with tf.device('/gpu:0'):
            # lookup生成张量
            embedding_inputs = tf.nn.embedding_lookup(self.embedding, self.input_x)

        with tf.name_scope("cnn"):
            # CNN layer
            conv = tf.layers.conv1d(embedding_inputs, self.config.filters_num, self.config.kernel_size, name='conv')
            # global max pooling layer
            gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')

        with tf.name_scope("score"):
            # 全连接层，后面接dropout以及relu激活
            fc = tf.layers.dense(gmp, self.config.hidden1_neuron_num, name='fc1')
            fc = tf.contrib.layers.dropout(fc, self.keep_prob)
            fc = tf.nn.relu(fc)

            # 分类器
            self.logits = tf.layers.dense(fc, self.config.classes_num, name='fc2')
            self.y_pred_cls = tf.argmax(tf.nn.softmax(self.logits), 1)  # 预测类别

        with tf.name_scope("optimize"):
            # 损失函数，交叉熵
            cross_entropy = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.input_y)
            self.loss = tf.reduce_mean(cross_entropy)
            # 优化器
            self.optim = tf.train.AdamOptimizer(learning_rate=self.config.learning_rate).minimize(self.loss)

        with tf.name_scope("accuracy"):
            # 准确率
            correct_pred = tf.equal(tf.argmax(self.input_y, 1), self.y_pred_cls)
            self.acc = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

def test():
    print('go!')
    pass


if __name__ == '__main__':
    test()