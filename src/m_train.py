#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
@File    :   m_train.py
@Time    :   2018/6/9 23:10
@Software:   PyCharm
@Author  :   Li Xiang
@Desc    :   process执行训练任务,部分数据生成后会保存中间结果，首次执行需要时间非常长
"""

import time
import datetime
import src.ProConfig
import src.features
import src.ModelCnn
import tensorflow as tf
import numpy as np
from src.PathConfig import PATH_CONFIG

__time__ = '2018/6/9 23:10'

def get_train_validation(x1, x2, x3, y, fen):
    ss = 2
    train_x1 = [x1[i] for i in range(len(x1)) if i % fen != ss]
    valid_x1 = [x1[i] for i in range(len(x1)) if i % fen == ss]
    train_x2 = [x2[i] for i in range(len(x2)) if i % fen != ss]
    valid_x2 = [x2[i] for i in range(len(x2)) if i % fen == ss]
    train_x3 = [x3[i] for i in range(len(x3)) if i % fen != ss]
    valid_x3 = [x3[i] for i in range(len(x3)) if i % fen == ss]
    train_y = [y[i] for i in range(len(y)) if i % fen != ss]
    valid_y = [y[i] for i in range(len(y)) if i % fen == ss]
    return train_x1, train_x2, train_x3, train_y, valid_x1, valid_x2, valid_x3, valid_y

def get_time_dif(start_time):
    """获取已使用时间"""
    end_time = time.time()
    time_dif = end_time - start_time
    return datetime.timedelta(seconds=int(round(time_dif)))


def batch_iter(x, x2, x3, y, batch_size=64):
    """生成批次数据"""
    data_len = len(x)
    num_batch = int((data_len - 1) / batch_size) + 1

    indices = np.random.permutation(np.arange(data_len))
    x_shuffle = np.array(x)[indices]
    x2_shuffle = np.array(x2)[indices]
    x3_shuffle = np.array(x3)[indices]
    y_shuffle = np.array(y)[indices]

    for i in range(num_batch):
        start_id = i * batch_size
        end_id = min((i + 1) * batch_size, data_len)
        yield x_shuffle[start_id:end_id], x2_shuffle[start_id:end_id], x3_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

def feed_data(model, x1_batch, x2_batch, x3_batch, y_batch, keep_prob, w2v_matrix, learn_rate):
    feed_dict = {
        model.input_voc_x: x1_batch,
        model.input_w2v_x: x2_batch,
        model.input_f2_x: x3_batch,
        model.input_y: y_batch,
        model.keep_prob: keep_prob,
        model.embedding_w2v: w2v_matrix,
        model.learning_rate: learn_rate
    }
    return feed_dict


def evaluate(sess, model, x1_, x2_, x3_, y_, w2v_matrix, learn_rate):
    """评估在某一数据上的准确率和损失"""
    data_len = len(x1_)
    batch_eval = batch_iter(x1_, x2_, x3_, y_, 128)
    total_loss = 0.0
    total_acc = 0.0
    for x1_batch, x2_batch, x3_batch, y_batch in batch_eval:
        batch_len = len(x1_batch)
        feed_dict = feed_data(model, x1_batch, x2_batch, x3_batch, y_batch, 1.0, w2v_matrix, learn_rate)
        loss, acc = sess.run([model.loss, model.acc], feed_dict=feed_dict)
        total_loss += loss * batch_len
        total_acc += acc * batch_len

    return total_loss / data_len, total_acc / data_len


def process():

    print('载入数据...')
    start_time = time.time()

    # 建立全局配置
    config = src.ProConfig.ProConfig()

    # 数据管理类
    f_mgr = src.features.Features(config)

    # 获得特征向量
    train_voc_x = f_mgr.voc_train_x()

    train_w2v_x = f_mgr.w2v_train_x()
    w2v_matrix = f_mgr.w2v_matrix()
    config.w2v_vocab_size = len(w2v_matrix)

    train_f2_x = f_mgr.f2_train_x()
    config.f2_dimension = len(train_f2_x[0])

    # 获得训练数据指标
    train_y = f_mgr.train_y()

    # 分割出部分验证集
    train_voc_x, train_w2v_x, train_f2_x, train_y, valid_voc_x, valid_w2v_x, valid_f2_x, valid_y= \
        get_train_validation(train_voc_x, train_w2v_x, train_f2_x, train_y, 155)

    time_dif = get_time_dif(start_time)
    print("数据载入完成:", time_dif)

    # 开始训练
    #model = src.ModelCnn.ModleCnn_1(config)
    model = src.ModelCnn.ModleCnn_3(config)

    # tensorboard配置
    tf.summary.scalar("loss", model.loss)
    tf.summary.scalar("accuracy", model.acc)
    merged_summary = tf.summary.merge_all()
    writer = tf.summary.FileWriter(PATH_CONFIG.get('dir_cnn_tb'))

    # 模型存储Saver
    saver = tf.train.Saver()

    # 创建session
    #session = tf.Session(config=tf.ConfigProto(log_device_placement=True))
    session = tf.Session()
    session.run(tf.global_variables_initializer())
    writer.add_graph(session.graph)

    print('开始执行训练...')
    start_time = time.time()
    total_batch = 0  # 总批次
    best_acc_val = 0.0  # 最佳验证集准确率
    best_acc_tra = 0.0
    best_los_tra = 0.0
    best_los_val = 0.0
    learning_rate= config.learning_rate
    last_improved = 0  # 记录上一次提升批次
    last_learn_rate_dec = 0
    require_improvement = 8000  # 如果超过1000轮未提升，提前结束训练
    learnrate_dec_rounds = 2000  # 如果超过1000轮未提升，提前结束训练

    flag = False
    for epoch in range(config.epochs_num):
        print('Epoch:', epoch + 1)
        batch_train = batch_iter(train_voc_x, train_w2v_x, train_f2_x, train_y, config.batch_size)
        for x1_batch, x2_batch, x3_batch, y_batch in batch_train:
            feed_dict = feed_data(model, x1_batch, x2_batch, x3_batch, y_batch, config.dropout_keep_prob, w2v_matrix, learning_rate)
            if total_batch % config.save_per_batch == 0:
                # 每多少轮次将训练结果写入tensorboard scalar
                s = session.run(merged_summary, feed_dict=feed_dict)
                writer.add_summary(s, total_batch)

            if total_batch % config.print_per_batch == 0:
                # 每多少轮次输出在训练集和验证集上的性能
                feed_dict[model.keep_prob] = 1.0
                loss_train, acc_train = session.run([model.loss, model.acc], feed_dict=feed_dict)
                loss_val, acc_val = evaluate(session, model, valid_voc_x, valid_w2v_x, valid_f2_x, valid_y, w2v_matrix, learning_rate)  # todo

                if acc_val > best_acc_val:
                    # 保存最好结果
                    best_acc_val = acc_val
                    best_acc_tra = acc_train
                    best_los_tra = loss_train
                    best_los_val = loss_val
                    last_improved = total_batch
                    saver.save(sess=session, save_path=PATH_CONFIG.get('file_cnn_best'))
                    improved_str = '*'
                elif acc_val == best_acc_val:
                    if (acc_train > best_acc_tra and loss_train <= best_los_tra and loss_val <= best_los_val) or \
                       (acc_train >= best_acc_tra and loss_train < best_los_tra and loss_val <= best_los_val) or \
                       (acc_train >= best_acc_tra and loss_train <= best_los_tra and loss_val < best_los_val):
                        # 保存最好结果
                        best_acc_val = acc_val
                        best_acc_tra = acc_train
                        best_los_tra = loss_train
                        best_los_val = loss_val
                        last_improved = total_batch
                        saver.save(sess=session, save_path=PATH_CONFIG.get('file_cnn_best'))
                        improved_str = '*'
                    else:
                        improved_str = ''
                else:
                    improved_str = ''

                time_dif = get_time_dif(start_time)
                msg = 'Iter: {0:>6}, Train Loss: {1:>6.2}, Train Acc: {2:>7.2%},' \
                      + ' Val Loss: {3:>6.2}, Val Acc: {4:>7.2%}, Time: {5} {6}'
                print(msg.format(total_batch, loss_train, acc_train, loss_val, acc_val, time_dif, improved_str))

            session.run(model.optim, feed_dict=feed_dict)  # 运行优化
            total_batch += 1

            if total_batch - last_improved > require_improvement:
                print("验证集长时间没有提升,强制训练结束...")
                flag = True
                break  # 跳出循环

            if total_batch - last_improved > learnrate_dec_rounds and total_batch - last_learn_rate_dec > learnrate_dec_rounds:
                learning_rate = max(learning_rate*config.learning_rate_dec_rate, config.learning_rate_min)
                last_learn_rate_dec = total_batch
                print("%d轮无结果,降低学习率%f"%(total_batch - last_improved, learning_rate))

        if flag:  # 同上
            break

    session.close()

if __name__ == '__main__':
    #test()
    process()

