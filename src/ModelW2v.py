#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    :   ModelW2v.py
@Time    :   2018/6/9 11:47
@Software:   PyCharm
@Author  :   Li Xiang
@Desc    :   一个w2v模型，提供对w2v的训练以及使用
"""

from gensim.models import word2vec
from src.PathConfig import PATH_CONFIG

__time__ = '2018/6/9 11:47'


class ModelW2v(object):
    """
    w2v类
    """

    def __init__(self, vec_size=None):
        self.__vec_size = vec_size
        self.__model = None
        self.load(vec_size)

        pass

    def train(self, lt_data, vec_size=None):
        if vec_size is not None:
            self.__vec_size = vec_size
        print('w2v开始训练: 数据量=%d , 词向量维度=%d' % (len(lt_data), self.__vec_size))
        self.__model = word2vec.Word2Vec(lt_data, size=self.__vec_size, workers=8)
        print('w2v训练完成,保存结果至文件...')
        self.save()

    def save(self):
        if self.__model is not None:
            self.__model.save(self.save_path())

    def load(self, vec_size):
        self.__vec_size = vec_size
        if self.__vec_size is not None:
            try:
                self.__model = word2vec.Word2Vec.load(self.save_path())
                print('w2v现有模块加载完成: %d' % self.__vec_size)
            except:
                self.__vec_size = None
                print('w2v现有模块加载失败')
        else:
            self.__model = None

    def is_mod_in(self):
        return self.__model is not None

    def save_path(self):
        return '%s/w2v_%d.mod' % (PATH_CONFIG.get('dir_w2v'), self.__vec_size)

    def get_index2word(self):
        return self.__model.wv.index2word

    def get_vectors(self):
        return self.__model.wv.vectors

def test():
    # 获得训练数据
    '''
    import numpy as np
    lt_train = cut_data.cut_train_data()
    print(len(lt_train))
    try:
        lt_train = np.concatenate((lt_train, cut_data.cut_test_data()))
        print(len(lt_train))
    except:
        print('没有测试数据可以用于训练!')
        pass

    aa = ModelW2v()
    aa.train(lt_train, 300)
    '''
    aa = ModelW2v(300)
    pass


if __name__ == '__main__':
    test()
