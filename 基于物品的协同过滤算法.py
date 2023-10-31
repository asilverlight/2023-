import pandas as pd
from math import pow, sqrt, log1p
import heapq
import random
from collections import defaultdict
from operator import itemgetter


# 加载数据集
def LoadMovieLensData(train_rate=0.1):
    """
    :param train_rate: 只选取一部分数据进行训练
    :return:预处理好的数据集
    """
    ratings = pd.read_csv(r'ml-latest-small\merged.csv')
    ratings = ratings[['userId', 'title']]  # 只取用户和电影两列
    train = []  # 训练数据集，只抽取一部分
    for idx, row in ratings.iterrows():
        user = str(row['userId'])
        item = (row['title'])
        if random.random() < train_rate:
            train.append([user, item])

    """
        建立User-Item表，结构如下：
            {"User1": {MovieID1, MoveID2, MoveID3,...}
             "User2": {MovieID12, MoveID5, MoveID8,...}
             ...
            }
        """

    trainData = dict()
    for user, item in train:
        trainData.setdefault(user, set())
        trainData[user].add(item)
    return trainData


train = LoadMovieLensData(0.08)
N = defaultdict(int)  # 记录每个物品的喜爱人数
itemSimilary = defaultdict(str)  # 相似度矩阵


def calculateMatrix():  # 计算相似度矩阵
    for user, items in train.items():
        for i in items:
            itemSimilary.setdefault(i, dict())
            N[i] += 1
            for j in items:
                if i == j:
                    continue
                itemSimilary[i].setdefault(j, 0)
                itemSimilary[i][j] += 1. / log1p(len(items) + 1.)
    for i, items in itemSimilary.items():
        mymax = 0
        for j, _ in items.items():
            itemSimilary[i][j] /= sqrt(N[i] * N[j])
            mymax = max(mymax, itemSimilary[i][j])
        itemSimilary[i] = {k: v / mymax for k, v in items.items()}


def recommend(user, N, K=10):
    """
    :param user: 用户ID
    :param N: 推荐商品个数
    :param K:某一商品有K个商品与它最相似
    :return:推荐结果
    """
    calculateMatrix()
    recommends = dict()
    # print(train[1])
    items = train[user]  # user的喜爱列表
    for item in items:
        temp = sorted(itemSimilary[item].items(), key=itemgetter(1),
                      reverse=True)[:min(K, len(items))]
        for i, sim in temp:
            if i in items:
                continue  # 如果与user喜爱的物品重复了，则直接跳过
            recommends.setdefault(i, 0.)
            recommends[i] += sim
    temp = dict(sorted(recommends.items(), key=itemgetter(1),
                       reverse=True)[:min(K, len(items))])
    result = temp.keys()
    return list(result)


print(recommend('1', 10, 10))
