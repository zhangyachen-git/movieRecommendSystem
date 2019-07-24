#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Date    : 2019-07-24 14:21:53
# @Author  : YC.ZH (aachen_z@163.com)
# @Link    : http://example.org
# @Version : $Id$

import pandas as pd

# 数据处理
ratings_df = pd.read_csv('../data/ml-latest-small/ratings.csv')
# 使用tail查看后几行数据，默认5行

movies_df = pd.read_csv('../data/ml-latest-small/movies.csv')

# 增加行号信息
movies_df['movieRow'] = movies_df.index

# 筛选movies_df中的特征
movies_df = movies_df[['movieRow', 'movieId', 'title']]
movies_df.to_csv('./output/movieProcessed.csv',
                 index=False, header=True, encoding='utf-8')

# 将ratings_df中的movieId替换为行号
ratings_df = pd.merge(ratings_df, movies_df, on='movieId')
# 使用head查看前几行数据，默认前5行

ratings_df = ratings_df[['userId', 'movieRow', 'rating']]
ratings_df.to_csv('./output/ratingsProcessed.csv',
                  index=False, header=True, encoding='utf-8')

