#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    :   m_predict.py
@Time    :   2018/6/9 23:11
@Software:   PyCharm
@Author  :   Li Xiang
@Desc    :   process方法直接生成test预测结果，并保存至文件
"""
import src.ModelCnn
import src.ProConfig
import src.features
import src.d_org_data
import codecs
import datetime
import tensorflow as tf

from src.PathConfig import PATH_CONFIG

__time__ = '2018/6/9 23:11'


# 预测
class CnnPredict(object):

    def __init__(self, f_mgr):
        self.f_mgr = f_mgr
        self.w2v_matrix = self.f_mgr.w2v_matrix()
        self.map_id_to_label = self.f_mgr.map_id_to_label()

        self.config = src.ProConfig.ProConfig()
        self.config.w2v_vocab_size = len(self.w2v_matrix)
        self.config.f2_dimension = len((self.f_mgr.f2_train_x())[0])
        self.model = src.ModelCnn.ModleCnn_3(self.config)
        self.session = tf.Session()
        self.session.run(tf.global_variables_initializer())
        saver = tf.train.Saver()
        saver.restore(sess=self.session, save_path=PATH_CONFIG.get('file_cnn_best'))  # 读取保存的模型

    def predict(self, data1, data2, data3):
        feed_dict = {
            self.model.input_voc_x: [data1],
            self.model.input_w2v_x: [data2],
            self.model.input_f2_x: [data3],
            self.model.keep_prob: 1.0,
            self.model.embedding_w2v: self.w2v_matrix
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return y_pred_cls[0]

    def predict_str(self, data, data2):
        return self.map_id2label(self.predict(data, data2))

    def predict_data(self, data1, data2, data3):
        feed_dict = {
            self.model.input_voc_x: data1,
            self.model.input_w2v_x: data2,
            self.model.input_f2_x: data3,
            self.model.keep_prob: 1.0,
            self.model.embedding_w2v: self.w2v_matrix
        }

        y_pred_cls = self.session.run(self.model.y_pred_cls, feed_dict=feed_dict)
        return list(y_pred_cls)

    def map_id2label(self, id):
        return self.map_id_to_label[id]


def process():
    # 创建数据管理类
    f_mgr = src.features.Features(src.ProConfig.ProConfig())

    # 创建工作模型类
    run = CnnPredict(f_mgr)

    print('加载测试数据...')
    test_voc_x = f_mgr.voc_test_x()
    test_w2v_x = f_mgr.w2v_test_x()
    test_f2_x = f_mgr.f2_test_x()

    print('加载测试数据id号 (从原始数据)...')
    test_id = src.d_org_data.org_test_index()

    # 生成过程
    print('开始生成结果...')
    data_len = len(test_w2v_x)
    batch_size = 20
    num_batch = int((data_len - 1) / batch_size) + 1
    lt_ids = []
    for i in range(num_batch):
        if int((i * 1.0 / num_batch) * 100) % 10 == 0:
            print("%d / %d" % (i, num_batch))
        bng_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)

        d1 = test_voc_x[bng_id:end_id]
        d2 = test_w2v_x[bng_id:end_id]
        d3 = test_f2_x[bng_id:end_id]

        lt_ids += run.predict_data(d1, d2, d3)

    print('结果存入文件...')
    lines = '\n'.join([str(tid) + ',' + run.map_id2label(id) for tid, id in zip(test_id, lt_ids)])

    #save_path = PATH_CONFIG.get('file_sub') + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'
    save_path = PATH_CONFIG.get('file_sub')
    with codecs.open(save_path, "w", "utf-8")as f:
        f.write(lines)
        f.write('\n')
    print('完成: ' + save_path)


if __name__ == '__main__':
    # test()
    process()
