#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    :   PathConfig.py
@Time    :   2018/6/9 11:32
@Software:   PyCharm
@Author  :   Li Xiang
@Desc    :   对文件路径进行集中管控
"""
import os
import src.ProConfig

__time__ = '2018/6/9 11:32'

class PathConfig(object):
    """
    路径配置类
    """
    def __init__(self, data_root=None):
        self.config = src.ProConfig.ProConfig()
        '''
        self.__version = 'c(%d)-h(%d-%d)-w2v(%d)-tfidf(%d)-sta-r(%f)' % (self.config.filters_num, \
                                                                         self.config.hidden1_neuron_num, \
                                                                         self.config.hidden2_neuron_num, \
                                                                         self.config.w2v_dimension, \
                                                                         self.config.tfidf_dimension, \
                                                                         self.config.learning_rate)
        
        self.__version = 'c(%d-%d)-f(%d)-fnn(%d-%d)-ker(%d)-w2v(%d)-tfidf(%d)-sta-r(%f)%s' % (self.config.filters_num, \
                                                                                                self.config.f1_hidden_neuron_num, \
                                                                                                self.config.f2_hidden_neuron_num, \
                                                                                                self.config.hidden1_neuron_num, \
                                                                                                self.config.hidden2_neuron_num, \
                                                                                                self.config.kernel_size,\
                                                                                                self.config.w2v_dimension, \
                                                                                                self.config.tfidf_dimension, \
                                                                                                self.config.learning_rate, \
    
        '''
        self.__version = 'voc(%d-%d-%d-%d-%d-%d)-w2v(%d-%d-%d-%d-%d-%d)-f2(%d)-fnn(%d-%d)-v(%s)' \
                         % ( self.config.filters_num_voc1, \
                             self.config.kernel_size_voc1,  \
                             self.config.filters_num_voc2,  \
                             self.config.kernel_size_voc2,  \
                             self.config.pool_size_voc,\
                             self.config.hidden_neuron_num_voc, \
 \
                             self.config.filters_num_w2v1, \
                             self.config.kernel_size_w2v1, \
                             self.config.filters_num_w2v2, \
                             self.config.kernel_size_w2v2, \
                             self.config.pool_size_w2v, \
                             self.config.hidden_neuron_num_w2v, \
 \
                             self.config.hidden_neuron_num_f2, \
 \
                             self.config.hidden_neuron_num_fin_1, \
                             self.config.hidden_neuron_num_fin_2, \
                             self.config.version)

        self.__path_dict = {}

        # 初始化基础路径
        if data_root is None:
            data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir, 'res'))
        self.__path_dict['dir_root'] = data_root

        # 原始数据路径
        self.__path_dict['dir_data'] = os.path.abspath(os.path.join(self.get('dir_root'), 'data'))
        self.make_sure_exist('dir_data')
        self.__path_dict['file_org_train'] = os.path.abspath(os.path.join(self.get('dir_data'), 'training.txt'))
        self.__path_dict['file_org_test'] = os.path.abspath(os.path.join(self.get('dir_data'), 'test.txt'))
        self.__path_dict['file_cut_train'] = os.path.abspath(os.path.join(self.get('dir_data'), 'cut_train.txt'))
        self.__path_dict['file_cut_test'] = os.path.abspath(os.path.join(self.get('dir_data'), 'cut_test.txt'))
        self.__path_dict['file_sta_train'] = os.path.abspath(os.path.join(self.get('dir_data'), 'sta_train.csv'))
        self.__path_dict['file_sta_test'] = os.path.abspath(os.path.join(self.get('dir_data'), 'sta_test.csv'))

        # 模型存储路径
        self.__path_dict['dir_mod'] = os.path.abspath(os.path.join(self.get('dir_root'), 'mod'))
        self.make_sure_exist('dir_mod')
        self.__path_dict['dir_w2v'] = os.path.abspath(os.path.join(self.get('dir_mod'), 'w2v'))
        self.make_sure_exist('dir_w2v')
        self.__path_dict['dir_voc'] = os.path.abspath(os.path.join(self.get('dir_mod'), 'voc'))
        self.make_sure_exist('dir_voc')
        self.__path_dict['dir_tfidt'] = os.path.abspath(os.path.join(self.get('dir_mod'), 'tfidf'))
        self.make_sure_exist('dir_tfidt')
        self.__path_dict['dir_cnn'] = os.path.abspath(os.path.join(self.get('dir_mod'), 'cnn'))
        self.make_sure_exist('dir_cnn')
        self.__path_dict['dir_cnn_cp'] = os.path.abspath(os.path.join(self.get('dir_cnn'), 'checkpoints', self.__version))
        self.make_sure_exist('dir_cnn_cp')
        self.__path_dict['dir_cnn_tb'] = os.path.abspath(os.path.join(self.get('dir_cnn'), 'tensorboard', self.__version))
        self.make_sure_exist('dir_cnn_tb')
        self.__path_dict['file_cnn_best'] = os.path.abspath(os.path.join(self.get('dir_cnn_cp'), 'best_validation'))

        # 提交列表
        self.__path_dict['dir_sub'] = os.path.abspath(os.path.join(self.get('dir_root'), 'sub'))
        self.make_sure_exist('dir_sub')
        # self.__path_dict['file_sub'] = os.path.abspath(os.path.join(self.get('dir_sub'), 'cnn_'))
        self.__path_dict['file_sub'] = os.path.abspath(os.path.join(self.get('dir_sub'), 'sub_1.csv'))
        self.__path_dict['file_sub_final'] = os.path.abspath(os.path.join(self.get('dir_sub'), 'cnn_'))

    def get(self,name):
        return self.__path_dict[name]

    def make_sure_exist(self, name):
        if not os.path.exists(self.get(name)):
            os.makedirs(self.get(name))


PATH_CONFIG = PathConfig()

def test():
    print(PATH_CONFIG.get('file_org_train'))
    pass

if __name__ == '__main__':
    test()