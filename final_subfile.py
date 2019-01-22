#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    :   final_subfile.py
@Time    :   2018/6/9 23:11
@Software:   PyCharm
@Author  :   Li Xiang
@Desc    :
"""

import datetime
import codecs
import src.PathConfig
import src2.PathConfig

PATHCONFIG_1 = src.PathConfig.PathConfig()
PATHCONFIG_2 = src2.PathConfig.PathConfig()

def process():
    file1 = PATHCONFIG_1.get('file_sub')
    file2 = PATHCONFIG_2.get('file_sub')

    with open(file1, 'r', encoding='utf-8', errors='ignore') as f:
        ltRet1 = [line for line in f]

    with open(file2, 'r', encoding='utf-8', errors='ignore') as f:
        ltRet2 = [line for line in f]

    ltRet = []
    for x1, x2 in zip(ltRet1, ltRet2):
        if ('人类作者' in x1) or ('自动摘要' in x1):
            ltRet.append(x2)
        else:
            ltRet.append(x1)

    save_path = PATHCONFIG_1.get('file_sub_final') + datetime.datetime.now().strftime('%Y%m%d_%H%M%S') + '.csv'
    with codecs.open(save_path, "w", "utf-8")as f:
        f.write("".join(ltRet))
    print('完成: ' + save_path)


if __name__ == '__main__':
    process()