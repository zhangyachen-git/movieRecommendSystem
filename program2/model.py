#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 14:32:56
# @Author  : YC.ZH (aachen_z@163.com)
# @Link    : http://example.org
# @Version : $Id$

import tensorflow as tf
import numpy as np
import dataPre


def normalizeRatings(rating, record):
    m, n = rating.shape
    rating_mean = np.zeros((m, 1))  # 初始化对于每部电影每个用户的平均评分
    rating_norm = np.zeros((m, n))  # 保存处理后的数据
    # 原始评分-平均评分，最后将计算结果和平均评分返回。
    for i in range(m):
        idx = record[i, :] != 0  # 获取已经评分的电影的下标
        rating_mean[i] = np.mean(rating[i, idx])  # 计算平均值，右边式子代表第i行已经评分的电影的平均值
        rating_norm[i, idx] = rating[i, idx] - rating_mean[i]
    return rating_norm, rating_mean


def model():
    rating, record, userNo, movieNo = dataPre.dataPre()
    print(rating, record, userNo, movieNo)
    rating_norm, rating_mean = normalizeRatings(rating, record)
    rating_norm = np.nan_to_num(rating_norm)
    rating_mean = np.nan_to_num(rating_mean)

    # 假设有10中类型的电影
    num_features = 10
    # 初始化电影矩阵X，用户喜好矩阵Theta,这里产生的参数都是随机数，并且是正态分布
    X_parameters = tf.Variable(tf.random_normal(
        [movieNo, num_features], stddev=0.35))
    Theta_paramters = tf.Variable(tf.random_normal(
        [userNo, num_features], stddev=0.35))
    # 理论课定义的代价函数
    # tf.matmul(X_parameters, Theta_paramters, transpose_b=True)代表X_parameters和Theta_paramters的转置相乘
    loss = 1 / 2 * tf.reduce_sum(((tf.matmul(X_parameters, Theta_paramters, transpose_b=True)
                                   - rating_norm) * record) ** 2) \
           + 1 / 2 * (tf.reduce_sum(X_parameters ** 2) +
                      tf.reduce_sum(Theta_paramters ** 2))  # 正则化项，其中λ=1，可以调整来观察模型性能变化。

    # 创建优化器和优化目标
    optimizer = tf.train.AdamOptimizer(1e-4)
    train = optimizer.minimize(loss)

    # Step4:训练模型
    # 使用TensorFlow中的tf.summary模块，它用于将TensorFlow的数据导出，从而变得可视化，因为loss是标量，所以使用scalar函数
    tf.summary.scalar('loss', loss)
    # 将所有summary信息汇总
    summaryMerged = tf.summary.merge_all()
    # 定义保存信息的路径
    filename = './output/movie_tensorboard'
    # 把信息保存在文件当中
    writer = tf.summary.FileWriter(filename)

    # 创建tensorflow绘画
    # Create a saver.
    saver = tf.train.Saver()
    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)
    # 开始训练模型
    for i in range(5001):
        _, movie_summary = sess.run([train, summaryMerged])
        writer.add_summary(movie_summary, i)
        if i % 500 == 0:
            # Append the step number to the checkpoint name:
            saver.save(sess, './model/my_model', global_step=i)
        # 记录一下次数，有没有都无所谓，只是看看还有多久
        print('i=', i, 'loss=', loss)
    # 评估模型
    Current_X_paramters, Current_Theta_parameters = sess.run(
        [X_parameters, Theta_paramters])
    # 将电影内容矩阵和用户喜好矩阵相乘，再加上每一行的均值，便得到一个完整的电影评分表
    predicts = np.dot(Current_X_paramters,
                      Current_Theta_parameters.T) + rating_mean
    # 计算预测值与真实值的残差平方和的算术平方根，将它作为误差error,随着迭代次数增加而减少
    errors = np.sqrt(np.sum((predicts - rating) ** 2))
    print('Current_X_paramters:', Current_X_paramters)
    print('Current_Theta_parameters:', Current_Theta_parameters)
    print('errors:', errors)


if __name__ == '__main__':
    model()
