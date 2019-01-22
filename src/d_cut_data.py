#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    :   d_cut_data.py
@Time    :   2018/6/9 12:35
@Software:   PyCharm
@Author  :   Li Xiang
@Desc    :   切分后数据提取接口
"""
import codecs
import sys

import jieba

import src.d_org_data
from src.PathConfig import PATH_CONFIG

__time__ = '2018/6/9 12:35'


def read_cut_file(filename):
    with open(filename, 'r', encoding='utf-8', errors='ignore') as f:
        ltTexts = [line for line in f]
    return ltTexts

def save_cut_file(filename, lt_cut_text):
    with codecs.open(filename, "w", "utf-8")as f:
        f.write('\n'.join(lt_cut_text))


def cut_train_text_label():
    lt_org_data, lt_label = src.d_org_data.org_train_data()
    try:
        lt_cut_text = read_cut_file(PATH_CONFIG.get('file_cut_train'))
    except:
        print('训练数据首次加载...')
        lt_cut_text = [' '.join(jieba.cut(text, cut_all=False)) for text in lt_org_data]
        lt_cut_text = [a.replace('\n', ' ').replace('\r', ' ') for a in lt_cut_text]
        # 顺手保存一下
        save_cut_file(PATH_CONFIG.get('file_cut_train'), lt_cut_text)

    return lt_cut_text, lt_label


def cut_train_text():
    try:
        lt_cut_text = read_cut_file(PATH_CONFIG.get('file_cut_train'))
    except:
        print('训练数据首次加载...')
        lt_org_data, _ = src.d_org_data.org_train_data()
        lt_cut_text = [' '.join(jieba.cut(text, cut_all=False)) for text in lt_org_data]
        lt_cut_text = [a.replace('\n', ' ').replace('\r', ' ') for a in lt_cut_text]
        # 顺手保存一下
        save_cut_file(PATH_CONFIG.get('file_cut_train'), lt_cut_text)

    return lt_cut_text


def cut_test_text():
    try:
        lt_cut_text = read_cut_file(PATH_CONFIG.get('file_cut_test'))
    except:
        print('测试数据首次加载...')
        lt_org_data = src.d_org_data.org_test_data()
        lt_cut_text = [' '.join(jieba.cut(text, cut_all=False)) for text in lt_org_data]
        lt_cut_text = [a.replace('\n', ' ').replace('\r', ' ') for a in lt_cut_text]
        # 顺手保存一下
        save_cut_file(PATH_CONFIG.get('file_cut_test'), lt_cut_text)

    return lt_cut_text


def cut_train_data_label():
    texts, lable = cut_train_text_label()
    return [x.split() for x in texts], lable


def cut_train_data():
    return [x.split() for x in cut_train_text()]


def cut_test_data():
    return [x.split() for x in cut_test_text()]


def test():
    cut = read_cut_file("E:\\competition\\SMP2018\\my_text_classification_cnn\\res\\data\\cut_train.txt")
    org, _ = src.d_org_data.org_train_data()
    for ii in range(len(org)):
        if cut[ii][0] != org[ii][0]:
            print(org[ii])
            print(cut[ii])
            break
'''
    a = cut_test_data()
    print(a[:10])
    pass
'''

if __name__ == '__main__':
    test()
