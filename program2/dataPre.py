#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 14:26:11
# @Author  : YC.ZH (aachen_z@163.com)
# @Link    : http://example.org
# @Version : $Id$

import pandas as pd
import numpy as np
import tensorflow as tf


def dataPre():
    ratings_df = pd.read_csv('./output/ratingsProcessed.csv')
    # 创建电影评分矩阵rating和评分记录矩阵record,即用户是否对电影进行了评分，评分则1，未评分则为0
    userNo = ratings_df['userId'].max() + 1
    movieNo = ratings_df['movieRow'].max() + 1

    rating = np.zeros((movieNo, userNo))

    flag = 0
    ratings_df_length = np.shape(ratings_df)[0]  # ratings_df的样本个数

    # 填写rating
    for index, row in ratings_df.iterrows():
        # 将rating当中对应的电影编号及用户编号填上row当中的评分
        rating[int(row['movieRow']), int(row['userId'])] = row['rating']
        flag += 1
        # print('processed %d, %d left' % (flag, ratings_df_length - flag))

    # 电影评分表中，>0表示已经评分，=0表示未被评分
    record = rating > 0
    # bool值转换为0和1
    record = np.array(record, dtype=int)
    return rating, record, userNo, movieNo
