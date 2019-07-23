#!/usr/bin/env python
# coding: utf-8


from math import pow, sqrt

# # 推荐系统

# ## 用字典存放所得数据

file = open('./data/merged.csv', 'r', encoding='utf-8')  # 记得读取文件时加‘r’， encoding='UTF-8'
# 读取data.csv中每行中除了名字的数据
data = {}  # 存放每位用户评论的电影和评分
for line in file.readlines():
    # 注意这里不是readline()
    print(line)
    line = line.strip().split(',')
    # 如果字典中没有某位用户，则使用用户ID来创建这位用户
    if not line[0] in data.keys():
        data[line[0]] = {line[3]: line[1]}
    # 否则直接添加以该用户ID为key字典中
    else:
        data[line[0]][line[3]] = line[1]


# ## 计算两个用户的相似度

# 注意：最后把距离缩放到了[0, 1]之间，这是为了简化计算。
# 因为有可能两个用户之间的差异很大，平方和累加起来是一个很大的数，他们两个差异这么大对这个推荐系统没用，所以用1/（1+distance）把它缩放到0.

def Euclidean(user1, user2):
    # 取出两位用户评论过的电影和评分
    user1_data = data[user1]
    user2_data = data[user2]
    distance = 0
    # 找到两位用户都评论过的电影，并计算欧式距离
    for key in user1_data.keys():
        if key in user2_data.keys():
            # 注意，distance越大表示两者越相似
            distance += pow(float(user1_data[key]) - float(user2_data[key]), 2)
            print(distance)
    return 1 / (1 + sqrt(distance))  # 这里返回值越大，相似度越大


# ## 找到最相似的k个用户
def top10_similar(userID):
    res = []
    for userid in data.keys():
        if not userid == userID:
            sim = Euclidean(userID, userid)
            print(sim)
            res.append((userid, sim))
    res.sort(key=lambda val: val[1], reverse=True)

    return res[:10]


# ## 找到最相似的用户看过的电影
def recommend(user, k=5):
    recomm = []
    most_sim_user = top10_similar(user)[0][0]
    items = data[most_sim_user]
    for item in items.keys():
        if item not in data[user].keys():
            recomm.append((item, items[item]))
    recomm.sort(key=lambda val: val[1], reverse=True)

    return recomm[:k]


if __name__ == '__main__':
    RES = top10_similar('1')
    print(RES)
    RECOM = recommend('1')
    print(RECOM)
