import model
import numpy as np
import pandas as pd
import dataPre
import tensorflow as tf


def run(user_id):
    # 调用训练好的模型
    with tf.Session() as sess:
        saver = tf.train.import_meta_graph('./model/my_model-5000.meta')
        saver.restore(sess, tf.train.latest_checkpoint('./model/'))
        movies_df = pd.read_csv('./output/movieProcessed.csv')
        rating, record, userNo, movieNo = dataPre.dataPre()
        rating_norm, rating_mean = model.normalizeRatings(rating, record)
        rating_norm = np.nan_to_num(rating_norm)
        rating_mean = np.nan_to_num(rating_mean)
        # 假设有10中类型的电影
        num_features = 10
        # 初始化电影矩阵X，用户喜好矩阵Theta,这里产生的参数都是随机数，并且是正态分布
        X_parameters = tf.Variable(tf.random_normal(
            [movieNo, num_features], stddev=0.35))
        Theta_paramters = tf.Variable(tf.random_normal(
            [userNo, num_features], stddev=0.35))
        # 初始化
        init = tf.global_variables_initializer()
        sess.run(init)
        Current_X_paramters, Current_Theta_parameters = sess.run(
            [X_parameters, Theta_paramters])
        predicts = np.dot(Current_X_paramters,
                          Current_Theta_parameters.T) + rating_mean
        # 获取对该用户的电影评分的列表，predicts[:, int(user_id)]是该用户对应的所有电影的评分，即系统预测的用户对于电影的评分
        # argsort()从小到大排序，argsort()[::-1]从大到小排序
        sortedResult = predicts[:, int(user_id)].argsort()[::-1]
        # idx用于保存已经推荐了多少部电影
        idx = 0
        print('为该用户推荐的评分最高的20部电影是：'.center(80, '='))
        for i in sortedResult:
            print('评分： %.2f, 电影名： %s' %
                  (predicts[i, int(user_id)], movies_df.iloc[i]['title']))
            idx += 1
            if idx == 20:
                break


if __name__ == '__main__':
    user_id = input('您要想哪位用户进行推荐？请输入用户编号：')
    run(user_id)
