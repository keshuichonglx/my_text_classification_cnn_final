#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    :   ProConfig.py
@Time    :   2018/6/9 23:16
@Software:   PyCharm
@Author  :   Li Xiang
@Desc    :   所有配置集中管控
"""

__time__ = '2018/6/9 23:16'


class ProConfig(object):
    def __init__(self):
        self.version = '0.2_r0.8_p'
        # voc统计工程
        self.voc_vocab_size = 3000  # voc字典大小
        self.voc_seq_length = 800   # voc长度
        self.voc_dimension = 300    # voc向量维度

        # cix统计工程
        self.cix_vocab_size = 0  # cix字典后期写入
        self.cix_seq_length = 500  # cix长度
        self.cix_dimension = 300  # cix向量维度维

        # w2v工程配置
        self.w2v_seq_length = 500   # w2v单文本矩阵高度
        self.w2v_vocab_size = 0     # w2v字典长度需要后期补入
        self.w2v_dimension = 300    # w2v向量维度

        # 统计工程配置
        self.tfidf_dimension = 3000  # tfidf向量维度
        self.f2_dimension = 0        # 统计类特征工程维度，毒药后期补入

        # 神经网络模型参数
        self.classes_num = 2        # 预测目标label数量

        # cix处理部分
        self.filters_num_cix1 = 1024  # 卷积核个数
        self.kernel_size_cix1 = 5  # 卷积核尺寸
        self.filters_num_cix2 = 256  # 卷积核个数
        self.kernel_size_cix2 = 5  # 卷积核尺
        self.pool_size_cix = 20
        self.hidden_neuron_num_cix = 256  # 全连接网络神经元个数

        # voc处理部分
        self.filters_num_voc1 = 1024  # 卷积核个数
        self.kernel_size_voc1 = 20   # 卷积核尺寸
        self.filters_num_voc2 = 256  # 卷积核个数
        self.kernel_size_voc2 = 10   # 卷积核尺
        self.pool_size_voc = 20
        self.hidden_neuron_num_voc = 1024  # 全连接网络神经元个数

        # w2v处理部分
        self.filters_num_w2v1 = 1024  # 卷积核个数
        self.kernel_size_w2v1 = 5   # 卷积核尺寸
        self.filters_num_w2v2 = 256  # 卷积核个数
        self.kernel_size_w2v2 = 5    # 卷积核尺
        self.pool_size_w2v = 20
        self.hidden_neuron_num_w2v = 256  # 全连接网络神经元个数

        # 统计工程处理部分
        self.hidden_neuron_num_f2 = 256   # 全连接网络神经元个数

        # 合成后处理部分
        self.hidden_neuron_num_fin_1 = 1024  # 2级全连接网络神经元个数 1层
        self.hidden_neuron_num_fin_2 = 0     # 2级全连接网络神经元个数 2层

        # 训练参数
        self.learning_rate = 0.0001       # 学习率
        self.learning_rate_min = 0.000001  # 最低学习率
        self.learning_rate_dec_rate = 0.8  # 学习率衰减
        self.dropout_keep_prob = 0.2  # dropout保留比例
        self.epochs_num = 10  # 总迭代轮次
        self.batch_size = 20  # 每批训练大小

        # 模型保存记录
        self.print_per_batch = 100  # 每多少轮输出一次结果
        self.save_per_batch = 50  # 每多少轮存入tensorboard

'''
0.989263697984025
{'人类作者': 0.9829044165772182, '机器作者': 0.9980325235896407, '自动摘要': 0.9812859884836853, '机器翻译': 0.9948318632855567}
class ProConfig(object):
    def __init__(self):
        self.version = '0.5_r0.9_vs5000'
        # voc统计工程
        self.voc_vocab_size = 5000  # voc字典大小
        self.voc_seq_length = 800   # voc长度
        self.voc_dimension = 300    # voc向量维度

        # w2v工程配置
        self.w2v_seq_length = 500   # w2v单文本矩阵高度
        self.w2v_vocab_size = 0     # w2v字典长度需要后期补入
        self.w2v_dimension = 300    # w2v向量维度

        # 统计工程配置
        self.tfidf_dimension = 3000  # tfidf向量维度
        self.f2_dimension = 0        # 统计类特征工程维度，毒药后期补入

        # 神经网络模型参数
        self.classes_num = 2        # 预测目标label数量

        # voc处理部分
        self.filters_num_voc1 = 512  # 卷积核个数
        self.kernel_size_voc1 = 20   # 卷积核尺寸
        self.filters_num_voc2 = 256  # 卷积核个数
        self.kernel_size_voc2 = 10   # 卷积核尺
        self.pool_size_voc = 20
        self.hidden_neuron_num_voc = 1500  # 全连接网络神经元个数

        # w2v处理部分
        self.filters_num_w2v1 = 512  # 卷积核个数
        self.kernel_size_w2v1 = 10   # 卷积核尺寸
        self.filters_num_w2v2 = 256  # 卷积核个数
        self.kernel_size_w2v2 = 5    # 卷积核尺
        self.pool_size_w2v = 20
        self.hidden_neuron_num_w2v = 512  # 全连接网络神经元个数

        # 统计工程处理部分
        self.hidden_neuron_num_f2 = 256   # 全连接网络神经元个数

        # 合成后处理部分
        self.hidden_neuron_num_fin_1 = 1024  # 2级全连接网络神经元个数 1层
        self.hidden_neuron_num_fin_2 = 0     # 2级全连接网络神经元个数 2层

        # 训练参数
        self.learning_rate = 0.0001       # 学习率
        self.learning_rate_min = 0.000001  # 最低学习率
        self.learning_rate_dec_rate = 0.9  # 学习率衰减
        self.dropout_keep_prob = 0.5  # dropout保留比例
        self.epochs_num = 10  # 总迭代轮次
        self.batch_size = 64  # 每批训练大小

        # 模型保存记录
        self.print_per_batch = 100  # 每多少轮输出一次结果
        self.save_per_batch = 50  # 每多少轮存入tensorboard
'''

def test():
    pass


if __name__ == '__main__':
    test()