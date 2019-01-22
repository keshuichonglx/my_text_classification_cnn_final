#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    :   m_run.py
@Time    :   2018/6/9 11:16
@Software:   PyCharm
@Author  :   Li Xiang
@Desc    :   中控调用
"""

__time__ = '2018/6/9 11:16'

import sys
import src2.m_train
import src2.m_predict


def main():
    if len(sys.argv) <= 1:
        show()
        exit()

    if sys.argv[1] == 'train':
        src2.m_train.process()
    elif sys.argv[1] == 'pre':
        src2.m_predict.process()
    else:
        show()
        exit()


def show():
    print( '请输入参数:\n'\
          +'  train: 对数据进行训练\n'\
          +'  pre:   对test数据进行测试')


if __name__ == '__main__':
    main()



