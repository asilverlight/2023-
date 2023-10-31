import pandas as pd
from math import pow, sqrt
import heapq

movies = pd.read_csv(r'ml-latest-small\movies.csv')
ratings = pd.read_csv(r'ml-latest-small\ratings.csv')
data = pd.merge(movies, ratings, on='movieId')
data[['userId', 'rating', 'movieId', 'title']].sort_values('userId'). \
    to_csv(r'ml-latest-small\merged.csv', index=False)
# 合并两个文件，按照用户id排序，包括用户id，评分，电影id，电影名
# 读取数据
file = open(r'ml-latest-small\merged.csv', 'r', encoding='UTF-8')
data = {}
file = file.readlines()
file = file[1:]  # 去除第一行表头
for myline in file:
    myline = myline.strip().split(',')
    if not myline[0] in data.keys():
        data[myline[0]] = {myline[3]: myline[1]}
    else:
        data[myline[0]][myline[3]] = myline[1]  # 只存储用户id，电影名，分数


def Euclidean(user1, user2):  # 计算欧式相似度
    user1_data = data[user1]
    user2_data = data[user2]
    distance = 0
    # 找到两位用户都评论过的电影，并计算欧式距离
    for key in user1_data.keys():
        if key in user2_data.keys():
            # 注意，distance越大表示两者越相似
            distance += pow(float(user1_data[key]) - float(user2_data[key]), 2)
    return '%.6f' % (1 / (1 + sqrt(distance)))


# 找最相似的k个用户
def topk_similar(userID, k):
    heap = []
    for key in data.keys():
        if userID != key:
            similarity = Euclidean(userID, key)
            temp = (similarity, key)  # 小根堆按第一个数排序
            heapq.heappush(heap, temp)
            if len(heap) > k:
                heapq.heappop(heap)
    heap.sort(reverse=True)
    return heap


# 找最相似k个用户看的电影
def recommend(user, k=10):
    recom = []
    most_sim_user = topk_similar(user, k)[0][1]
    items = data[most_sim_user]
    for item in items.keys():
        if item not in data[user].keys():
            recom.append((item, items[item]))
    recom.sort(key=lambda val: val[1], reverse=True)
    newrecom = []
    for i in range(len(recom)):
        if float(recom[i][1]) >= 5:
            newrecom.append(recom[i][0])
    return newrecom


print(recommend('1', 10))
