#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    :   ModelTfidf.py
@Time    :   2018/6/11 0:06
@Software:   PyCharm
@Author  :   Li Xiang
@Desc    :   tfidf模型管理
"""

from sklearn.externals import joblib
from src.PathConfig import PATH_CONFIG
from sklearn.feature_extraction.text import TfidfVectorizer

__time__ = '2018/6/10 23:55'


class ModelTfidf(object):
    def __init__(self, vec_size=None):
        self.__vec_size = vec_size
        self.__model = None
        self.load(vec_size)

        pass

    def load(self, vec_size):
        self.__vec_size = vec_size
        if self.__vec_size is not None:
            try:
                self.__model = joblib.load(self.save_path())
                print('Tfidf现有模块加载完成: %d' % self.__vec_size)
            except:
                self.__vec_size = None
                print('Tfidf现有模块加载失败')
        else:
            self.__model = None

    def train(self, lt_data, vec_size=None):
        if vec_size is not None:
            self.__vec_size = vec_size
        print('Tfidf开始训练: 数据量=%d , 词向量维度=%d' % (len(lt_data), self.__vec_size))
        self.__model = TfidfVectorizer(stop_words=None, max_features=self.__vec_size)
        self.__model.fit_transform(lt_data)
        print('Tfidf训练完成,保存结果至文件...')
        self.save()

    def save(self):
        if self.__model is not None:
            joblib.dump(self.__model, self.save_path())


    def save_path(self):
        return '%s/tfidf_%d.mod' % (PATH_CONFIG.get('dir_tfidt'), self.__vec_size)


    def is_mod_in(self):
        return self.__model is not None


    def mod(self):
        return self.__model


def test():

    aaa = ModelTfidf(2000)
    if not aaa.is_mod_in():
        import numpy as np
        import d_cut_data
        lt_train = d_cut_data.cut_train_text()
        print(len(lt_train))
        try:
            lt_test = d_cut_data.cut_test_text()
            print(len(lt_test))
            lt_train = lt_train + lt_test
            print(len(lt_train))
        except:
            print('没有测试数据可以用于训练!')
            pass
        aaa.train(lt_train, 2000)
    pass


if __name__ == '__main__':
    test()





