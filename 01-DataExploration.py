#!/usr/bin/env python
# coding: utf-8

# # 电影推荐系统python实现

import pandas as pd

# ## 数据观察

movies_df = pd.read_csv('./data/ml-latest-small/movies.csv')
print("movies_df")
print(movies_df.head())

links_df = pd.read_csv('./data/ml-latest-small/links.csv')
print("links_df")
print(links_df.head())

ratings_df = pd.read_csv('./data/ml-latest-small/ratings.csv')
print("ratings_df")
print(ratings_df.head())

tags_df = pd.read_csv('./data/ml-latest-small/tags.csv')
print("tags_df")
print(tags_df.head())


# ## 数据处理

# ### 数据合并

# 目的是给定一个用户id，找出用户可能喜欢的电影名。
# 但是两个文件电影信息和用户评分信息是分开的，所以需要合并。
data = pd.merge(movies_df, ratings_df, on='movieId')  # 通过两数据框之间的movieId连接
data[['userId', 'rating', 'movieId', 'title']].sort_values(
    'userId').to_csv('./data/merged.csv', index=False)
