#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    :   features.py
@Time    :   2018/6/10 15:23
@Software:   PyCharm
@Author  :   Li Xiang
@Desc    :   特征工程管理类
"""
import jieba.posseg
import numpy as np
import pandas as pd
import tensorflow.contrib.keras as kr

from collections import Counter

import src2.ModelTfidf
import src2.ModelW2v
import src2.d_cut_data
import src2.d_org_data

from src2.PathConfig import PATH_CONFIG

__time__ = '2018/6/10 15:23'


class Features(object):

    def __init__(self, config):
        self.config = config

        self.__map_label_to_id = {'自动摘要': 0, '人类作者': 1}
        self.__map_id_to_label = {0: '自动摘要', 1: '人类作者'}

        self.__org_texts = None
        self.__org_labels = None

        self.__train_texts = None
        self.__test_texts = None
        self.__train_datas = None
        self.__test_datas = None
        self.__train_org_texts = None
        self.__test_org_texts = None

        self.__w2v_matrix = None
        self.__w2v_word2id = None

        self.__w2v_train_x = None
        self.__w2v_test_x = None

        self.__voc_train_x = None
        self.__voc_test_x = None
        self.__voc_word2id = None

        self.__tfidf_mod = None
        self.__tfidf_train_x = None
        self.__tfidf_test_x = None

        self.__sta_train_x = None
        self.__sta_test_x = None

        self.__f2_train_x = None
        self.__f2_test_x = None

        self.__train_y = None

        self.__voc_train_2_x = None
        self.__cix_train_2_x = None
        self.__w2v_train_2_x = None
        self.__f2_train_2_x = None
        self.__train_2_y = None

        self.__sta_ltCix = [
            's', 'g', 'nr', 'nt', 'yg', 'e', 'vi', 'r', 'rz', 'ug', 'rr', 'c', 'm', 'b', 'vn', 'o', 'nrt',
            'z', 'f', 'eng', 'vg', 'a', 'j', 'uj', 'nz', 'k', 'v', 'zg', 'd', 'uz', 'ng', 'ul', 'mq', 'y',
            'h', 'dg', 'vd', 'nrfg', 'uv', 'ad', 'n', 'ns', 'i', 'x', 'ud', 'u', 'ag', 'q', 'an', 'vq', 'p',
            'df', 'rg', 't', 'tg', 'l'
        ]
        self.__sta_ltPun = [
            ' ', '!', '"', '#', '%', '&', "'", '(', ')', '*', '+', ',', '-', '.', '/', ':', ';', '<',
            '=', '>', '?', '[', ']', '_', '{', '|', '}', '‘', '’', '“', '”', '…', '、', '。', '《', '》',
            '【', '】', '！', '＃', '％', '＆', '（', '）', '＋', '，', '－', '．', '／', '：', '；', '＝',
            '？', '｜', '～'
        ]
        self.__sta_ltdfCol = ['len'] + self.__sta_ltCix + self.__sta_ltPun

        self.__cix_word2cix = jieba.posseg.dt.word_tag_tab
        self.__cix_cix2id = {}
        for i, cix in enumerate(self.__sta_ltCix):
            self.__cix_cix2id[cix] = i

        self.cix_vocab_size = len(self.__sta_ltCix)

        self.__cix_train_x = None
        self.__cix_test_x = None
        ###################################################################################
        self.__sta_ltPun += self.voc_word2id().keys()
        ###################################################################################


    def cix_train_x(self):
        if self.__cix_train_x is None:
            contents = self.train_datas()
            data_id = []
            for i in range(len(contents)):
                data_id.append([self.cix_word2id(x) for x in contents[i] if x in (self.__cix_word2cix.keys())])
            self.__cix_train_x = kr.preprocessing.sequence.pad_sequences(data_id, self.config.cix_seq_length)
        return self.__cix_train_x

    def cix_test_x(self):
        if self.__cix_test_x is None:
            contents = self.test_datas()
            data_id = []
            for i in range(len(contents)):
                data_id.append([self.cix_word2id(x) for x in contents[i] if x in (self.__cix_word2cix.keys())])
            self.__cix_test_x = kr.preprocessing.sequence.pad_sequences(data_id, self.config.cix_seq_length)
        return self.__cix_test_x

    def cix_word2id(self, word):
        try:
            return self.__cix_cix2id[self.cix_word2cix(word)]
        except:
            return self.__cix_cix2id['x']

    def cix_word2cix(self, word):
        try:
            return self.__cix_word2cix[word]
        except:
            return 'x'

    def voc_train_x(self):
        if self.__voc_train_x is None:
            contents = self.train_org_texts()
            data_id = []
            for i in range(len(contents)):
                data_id.append([(self.voc_word2id())[x] for x in contents[i] if x in (self.voc_word2id())])
            self.__voc_train_x = kr.preprocessing.sequence.pad_sequences(data_id, self.config.voc_seq_length)
        return self.__voc_train_x

    def voc_test_x(self):
        if self.__voc_test_x is None:
            contents = self.test_org_texts()
            data_id = []
            for i in range(len(contents)):
                data_id.append([(self.voc_word2id())[x] for x in contents[i] if x in (self.voc_word2id())])
            self.__voc_test_x = kr.preprocessing.sequence.pad_sequences(data_id, self.config.voc_seq_length)
        return self.__voc_test_x

    def voc_word2id(self):
        if self.__voc_word2id is None:
            self.load_voc_mod()
        return self.__voc_word2id

    def load_voc_mod(self):
        #lt_train = src2.d_cut_data.cut_train_text()
        #lt_train, _ = src2.d_org_data.org_train_data()
        lt_train = self.train_org_texts()
        print('得到训练数据 %d 条...' % len(lt_train))
        try:
            #lt_test = src2.d_cut_data.cut_test_text()
            #lt_test = src2.d_org_data.org_test_data()
            lt_test = self.test_org_texts()
            lt_train = lt_train + lt_test
            print('得到测试数据 %d 条...' % len(lt_test))
        except:
            print('没有测试数据可以用于训练!')
            pass

        all_data = []
        for content in lt_train:
            all_data.extend(content)

        counter = Counter(all_data)
        count_pairs = counter.most_common(self.config.voc_vocab_size - 1)
        words, _ = list(zip(*count_pairs))
        # 站位符
        words = ['<PAD>'] + list(words)

        word_to_id = dict(zip(words, range(len(words))))

        for key in word_to_id.keys():
            word_to_id[key] += 1

        self.__voc_word2id = word_to_id
    
    def create_sta_data(self, texts, filename):
        texts_size = len(texts)
        # 先创建一个pandas
        df = pd.DataFrame(index=range(texts_size), columns=self.__sta_ltdfCol).fillna(0)

        # 最先存入长度
        df['len'] = [len(x) for x in texts]

        # 符号统计
        print('符号统计...')
        for wp in self.__sta_ltPun:
            df[wp] = [x.count(wp) for x in texts]
            df[wp] = df[wp] * 100.0 / df['len']

        # 最慢的词性统计
        print('词性统计...')
        dicVoc = {}
        for a in self.__sta_ltCix:
            dicVoc[a] = [0] * texts_size

        for i in range(texts_size):
            if i % 5000 == 0:
                print('%d / %d' % (i, texts_size))
            # 分词，附带词性获取
            for w in jieba.posseg.cut(texts[i]):
                try:
                    dicVoc[w.flag][i] += 1
                except:
                    pass

        # 整列写入
        for a in self.__sta_ltCix:
            df[a] = np.array(dicVoc[a]) * 100.0 / df['len']

        # 最后稍微降低一下len的数值
        df['len'] = df['len'] * 1.0 / 600

        # 做完了,存文件
        df.to_csv(filename)
        return df[self.__sta_ltdfCol].values

    def sta_train_x(self):
        if self.__sta_train_x is None:
            # 从文件加载
            try:
                df = pd.read_csv(PATH_CONFIG.get('file_sta_train'))
                self.__sta_train_x = df[self.__sta_ltdfCol].values
            except:
                print('重新计算统计数据 - train')
                #texts, _ = src2.d_org_data.org_train_data()
                texts = self.train_org_texts()
                self.__sta_train_x = self.create_sta_data(texts, PATH_CONFIG.get('file_sta_train'))
        return self.__sta_train_x

    def sta_test_x(self):
        if self.__sta_test_x is None:
            # 从文件加载
            try:
                df = pd.read_csv(PATH_CONFIG.get('file_sta_test'))
                self.__sta_test_x = df[self.__sta_ltdfCol].values
            except:
                print('重新计算统计数据 - test')
                #texts = src2.d_org_data.org_test_data()
                texts = self.test_org_texts()
                self.__sta_test_x = self.create_sta_data(texts, PATH_CONFIG.get('file_sta_test'))
        return self.__sta_test_x

    def load_tfidf_mod(self):
        mod = src2.ModelTfidf.ModelTfidf(self.config.tfidf_dimension)
        if not mod.is_mod_in():
            #lt_train = src2.d_cut_data.cut_train_text()
            lt_train = self.train_texts()
            print('得到训练数据 %d 条...' % len(lt_train))
            try:
                #lt_test = src2.d_cut_data.cut_test_text()
                lt_test = self.test_texts()
                lt_train = lt_train + lt_test
                print('得到测试数据 %d 条...' % len(lt_test))
            except:
                print('没有测试数据可以用于训练!')
                pass

            if len(lt_train) <= 0:
                print('训练数据准备不足!!!')
                exit()
            mod.train(lt_train, self.config.tfidf_dimension)

        self.__tfidf_mod = mod.mod()


    def tfidf_train_x(self):
        if self.__tfidf_train_x is None:
            print('加载tfidf_train...')
            if self.__tfidf_mod is None:
                self.load_tfidf_mod()
            self.__tfidf_train_x = self.__tfidf_mod.transform(self.train_texts()).toarray()
        return self.__tfidf_train_x

    def tfidf_test_x(self):
        if self.__tfidf_test_x is None:
            print('加载tfidf_test...')
            if self.__tfidf_mod is None:
                self.load_tfidf_mod()
            self.__tfidf_test_x = self.__tfidf_mod.transform(self.test_texts()).toarray()
        return self.__tfidf_test_x


    def labels(self):
        if self.__org_labels is None:
            print('加载labels...')
            self.__train_org_texts, self.__org_labels = src2.d_org_data.org_train_data()
        return self.__org_labels
    '''
    def train_y(self):
        if self.__train_y is None:
            print('加载train_y...')
            self.__train_y = kr.utils.to_categorical([self.__map_label_to_id[x] for x in self.labels()],
                                                     num_classes=self.config.classes_num)
        return self.__train_y
    '''

    def train_2_y(self):
        if self.__train_2_y is None:
            train_2_y = [self.__map_label_to_id[label] for label in self.labels() if label in self.__map_label_to_id.keys()]
            self.__train_2_y = kr.utils.to_categorical(train_2_y, num_classes=self.config.classes_num)
        return self.__train_2_y

    def f2_train_2_x(self):
        if self.__f2_train_2_x is None:
            self.__f2_train_2_x = np.array([x for x, label in zip(self.f2_train_x(), self.labels()) \
                                   if label in self.__map_label_to_id.keys()])
        return self.__f2_train_2_x

    def w2v_train_2_x(self):
        if self.__w2v_train_2_x is None:
            self.__w2v_train_2_x = np.array([x for x, label in zip(self.w2v_train_x(), self.labels()) \
                                   if label in self.__map_label_to_id.keys()])
        return self.__w2v_train_2_x

    def voc_train_2_x(self):
        if self.__voc_train_2_x is None:
            self.__voc_train_2_x = np.array([x for x, label in zip(self.voc_train_x(), self.labels()) \
                                   if label in self.__map_label_to_id.keys()])
        return self.__voc_train_2_x

    def cix_train_2_x(self):
        if self.__cix_train_2_x is None:
            self.__cix_train_2_x = np.array([x for x, label in zip(self.cix_train_x(), self.labels()) \
                                   if label in self.__map_label_to_id.keys()])
        return self.__cix_train_2_x

    def f2_train_x(self):
        if self.__f2_train_x is None:
            print('加载f2_train...')
            self.__f2_train_x = np.array([np.append((self.tfidf_train_x())[i], (self.sta_train_x())[i]) for i in
                                          range(len(self.tfidf_train_x()))])
            self.__f2_train_x = np.nan_to_num(self.__f2_train_x)
        return self.__f2_train_x
    
    def w2v_train_x(self):
        if self.__w2v_train_x is None:
            print('加载w2v_train_x...')
            self.__w2v_train_x = self.get_w2v_vecs(self.train_datas(), self.w2v_word2id())
        return self.__w2v_train_x

    def f2_test_x(self):
        if self.__f2_test_x is None:
            print('加载f2_test...')
            self.__f2_test_x = np.array([np.append((self.tfidf_test_x())[i], (self.sta_test_x())[i]) for i in
                                          range(len(self.tfidf_test_x()))])
            self.__f2_test_x = np.nan_to_num(self.__f2_test_x)
        return self.__f2_test_x

    def w2v_test_x(self):
        if self.__w2v_test_x is None:
            print('加载w2v_test_x...')
            self.__w2v_test_x = self.get_w2v_vecs(self.test_datas(), self.w2v_word2id())
        return self.__w2v_test_x

    def train_texts(self):
        if self.__train_texts is None:
            self.__train_texts = src2.d_cut_data.cut_train_text()
        return self.__train_texts

    def test_texts(self):
        if self.__test_texts is None:
            self.__test_texts = src2.d_cut_data.cut_test_text()
        return self.__test_texts

    def train_org_texts(self):
        if self.__train_org_texts is None:
            self.__train_org_texts, self.__org_labels = src2.d_org_data.org_train_data()
        return self.__train_org_texts

    def test_org_texts(self):
        if self.__test_org_texts is None:
            self.__test_org_texts = src2.d_org_data.org_test_data()
        return self.__test_org_texts

    def train_datas(self):
        if self.__train_datas is None:
            self.__train_datas = src2.d_cut_data.cut_train_data()
        return self.__train_datas

    def test_datas(self):
        if self.__test_datas is None:
            self.__test_datas = src2.d_cut_data.cut_test_data()
        return self.__test_datas

    def w2v_matrix(self):
        if self.__w2v_matrix is None:
            self.load_w2v_mod()
        return self.__w2v_matrix

    def w2v_word2id(self):
        if self.__w2v_word2id is None:
            self.load_w2v_mod()
        return self.__w2v_word2id

    def load_w2v_mod(self):
        """训练w2v模型,从中得到词汇表和对应的向量矩阵"""
        # 获得模型
        print('加载w2v模型 - (%d)' % (self.config.w2v_dimension))
        mod = src2.ModelW2v.ModelW2v(self.config.w2v_dimension)
        if not mod.is_mod_in():
            print('加载w2v模型失败,重新训练一个模型....')

            lt_train = src2.d_cut_data.cut_train_data()
            print('得到训练数据 %d 条...' % len(lt_train))
            try:
                lt_test = src2.d_cut_data.cut_test_data()
                lt_train = np.concatenate((lt_train, lt_test))
                print('得到测试数据 %d 条...' % len(lt_test))
            except:
                print('没有测试数据可以用于训练!')
                pass

            if len(lt_train) <= 0:
                print('训练数据准备不足!!!')
                exit()
            mod.train(lt_train, self.config.w2v_dimension)

        # 获得矩阵和词列表
        word_to_id = dict(zip(mod.get_index2word(), range(len(mod.get_index2word()))))
        matrix = mod.get_vectors()

        matrix = np.pad(matrix, ((1, 0), (0, 0)), 'constant')
        for key in word_to_id.keys():
            word_to_id[key] += 1

        self.__w2v_matrix = matrix
        self.__w2v_word2id = word_to_id


    def get_w2v_vecs(self, cut_data, word2id):
        ret = [[word2id[s] for s in data if s in word2id] for data in cut_data]
        # 检查一遍有没有出现空行，补一个0，避免被消除
        for ll in range(len(ret)):
            if len(ret[ll]) <= 0:
                ret[ll] = [0]
        return kr.preprocessing.sequence.pad_sequences(ret, self.config.w2v_seq_length)

    def map_id_to_label(self):
        return self.__map_id_to_label


def test():

    import src2.ProConfig
    aa = Features(src2.ProConfig.ProConfig())
    print(aa.cix_word2id('黄晓明'))
    print(aa.cix_word2id('古天乐'))
    print(aa.cix_word2id('足球'))
    print(aa.cix_word2id('跳楼'))
    '''
    print(len(aa.f2_train_2_x()))
    print(len(aa.w2v_train_2_x()))

    
    print((aa.sta_train_x())[0])
    print((aa.sta_test_x())[1])
    print(len(aa.sta_train_x()))
    print(len(aa.sta_test_x()))
    '''


if __name__ == '__main__':
    test()
