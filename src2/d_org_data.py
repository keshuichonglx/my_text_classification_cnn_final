#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    :   d_org_data.py
@Time    :   2018/6/9 11:49
@Software:   PyCharm
@Author  :   Li Xiang
@Desc    :   原始数据提取接口
"""

import pandas as pd
from src2.PathConfig import PATH_CONFIG

__time__ = '2018/6/9 11:49'


def org_train_data():
    df = pd.read_json(PATH_CONFIG.get('file_org_train'), lines=True).set_index('id')
    return list(df['内容'].values), list(df['标签'].values)


def org_test_data():
    df = pd.read_json(PATH_CONFIG.get('file_org_test'), lines=True).set_index('id')
    return list(df['内容'].values)


def org_test_index():
    df = pd.read_json(PATH_CONFIG.get('file_org_test'), lines=True).set_index('id')
    return list(df.index.values)


def test():
    lt_text, lt_label = org_train_data()
    print(lt_text[10:20])
    print(lt_label[10:20])

    lt_text = org_test_data()
    print(lt_text[10:20])
    pass


if __name__ == '__main__':
    test()
