import pandas as pd
from math import pow, sqrt, log1p
import heapq
import random
from collections import defaultdict
from operator import itemgetter


class ItemCF:
    def __init__(self, user, training_rate=1):
        self.user = user
        self.train_rate = training_rate
        self.train = []  # 总数据集，包含用户、评分、电影
        self.items_has_seen_high = []  # 用户已经看过的高分电影
        self.items_has_seen = []  # 用户已经看过的所有电影
        self.item_user = dict()  # item对user矩阵
        self.result = dict()  # 存储最终筛选出的电影以及得分：
        self.user_has_seen = dict()  # 用户看过的电影以及其评分
        self.similarity_matrix = dict()  # 相似矩阵
        """
        result:
        {movie1:score1,
        movie2:score2}
        """

    def LoadMovieLensData(self):
        """
        :param train_rate: 只选取一部分数据进行训练
        :return:预处理好的数据集
        """
        ratings = pd.read_csv(r'ml-latest-small\merged.csv')
        ratings = ratings[['userId', 'rating', 'title']]
        train = []  # 训练数据集，只抽取一部分
        for idx, row in ratings.iterrows():
            user = str(row['userId'])
            rating = int(row['rating'])
            item = (row['title'])
            if random.random() < self.train_rate:
                train.append([user, rating, item])

        """
            建立User-Item表，结构如下：
                {"User1": {MovieID1, MoveID2, MoveID3,...}
                 "User2": {MovieID12, MoveID5, MoveID8,...}
                 ...
                }
            """

        trainData = dict()
        for user, rating, item in train:
            trainData.setdefault(user, set())
            trainData[user].add((rating, item))
            self.item_user.setdefault(item, set())
            self.item_user[item].add((user, rating))
        self.train = trainData
        """
        self.train:
        {
        user1:{(rating,item),(rating,item)},
        user2:{(rating,item),(rating,item)}
        }
        self.item_user:
        {
        item1:{(user,rating),(user,rating)},
        item2:{(user,rating),(user,rating)}
        }
        """
        """
        self.user_has_seen:
        {'Fantasia': 5, 'Winnie': 5, 'Alice': 5}
        """
        self.items_has_seen = self.train[self.user]
        temp = []
        for item in self.items_has_seen:
            self.user_has_seen.setdefault(item[1], item[0])
            temp.append(item[1])
        self.items_has_seen = temp  # list类型

    def find_high_item(self, M=10):  # 找某一用户前M个最高分商品
        items = self.train[self.user]
        M = min(M, len(items))
        items = sorted(items, key=itemgetter(0), reverse=True)[:M]
        # 至此找到用户user看过的前M个最高分电影，以set存储：
        # [(5, 'Talented Mr. Ripley, The (1999)'), (4, 'Toys (1992)')]
        self.items_has_seen_high = items  # 返回高分商品

    def cal_weight(self, item1, item2):  # 计算相似性
        item1_dict = sorted(self.item_user[item1], key=itemgetter(0))
        item2_dict = sorted(self.item_user[item2], key=itemgetter(0))
        # [('1', 5), ('103', 4), ('116', 4), ('128', 4)]
        # 两列物品字典按照用户名称升序排列
        # 采用双指针技术计算二者余弦距离
        len1 = len(item1_dict)
        len2 = len(item2_dict)
        p1 = 0  # 指针1
        p2 = 0  # 指针2
        up = 0  # 分子
        down1 = 0
        down2 = 0  # 分母
        while p1 < len1 and p2 < len2:
            if int(item1_dict[p1][0]) == int(item2_dict[p2][0]):  # 该电影有同样的人看过
                down1 += item1_dict[p1][1] ** 2
                down2 += item2_dict[p2][1] ** 2
                up += item1_dict[p1][1] * item2_dict[p2][1]
                p1 += 1
                p2 += 1
                continue
            if int(item1_dict[p1][0]) < int(item2_dict[p2][0]):
                # item2对应用户标号大，则p1前移
                down1 += item1_dict[p1][1] ** 2
                p1 += 1
                continue
            # item1对应用户标号大，则p2前移
            down2 += item2_dict[p2][1] ** 2
            p2 += 1
        if p1 != len1 and p2 == len2:  # 第二个item到头，接着计算第一个
            while p1 < len1:
                down1 += item1_dict[p1][1] ** 2
                p1 += 1
        if p2 != len2 and p1 == len1:  # 第一个item到头，接着计算第二个
            while p2 < len2:
                down2 += item2_dict[p2][1] ** 2
                p2 += 1
        cos_similarity = 0
        if sqrt(down1 * down2) != 0:
            cos_similarity = up / sqrt(down1 * down2)  # 余弦相似度
        return cos_similarity

    def find_havnt_seen(self):  # 针对每个看过的电影，找N个与其相似的、高分的、没看过的电影
        items_high = [v[1] for v in self.items_has_seen_high]  # user看过的高分电影
        items_has_seen = self.items_has_seen  # user看过的电影
        # 接下来，要找与高分电影相似度高的，且user没看过的电影
        # 理想存储形式：
        """
        self.similar_item
        {
        item1:{item11:5,item12:4}
        item2:{item21:5,item22:4}
        }
        item1,item2是未看过的电影,item11,item12是已看过的电影
        """
        # 需要存储item与其他所有user没看过的电影的相似度，并取出最高的N个电影
        # print(items_high)['L.A. Confidential (1997)', 'Seven (a.k.a. Se7en) (1995)',
        # 'Lord of the Rings, The (1978)']
        for item1 in items_high:  # user看过的高分电影
            for item2 in self.item_user.keys():  # 其他user没看过的电影
                # print(item1,item2)
                if item2 != item1 and (item2 not in items_has_seen):  # 该电影没看过
                    cos_sim = self.cal_weight(item1, item2)
                    self.similarity_matrix.setdefault(item2, set())
                    self.similarity_matrix[item2].add((item1, cos_sim))

    def cal_result(self, N=5, num=20):
        # 接下来，针对每个未看过的新电影，只取与其相似的N个电影，最终推荐num个电影
        self.result = []
        for new_movie in self.similarity_matrix.keys():
            self.similarity_matrix[new_movie] = sorted(self.similarity_matrix[new_movie],
                                                       key=itemgetter(1),
                                                       reverse=True)[:min(N, len(self.similarity_matrix[new_movie]))]
            # 接下来计算new movie的最终得分
            """
            similarity_matrix[new_movie]:这里的电影都是user看过的
            [('Lock, Stock', 0.208), ('Clockwork', 0.153)]
            """
            sim_len = len(self.similarity_matrix[new_movie])
            final_score = 0
            for i in range(sim_len):  # 评分×相关度/总相关度，再相加
                final_score += self.user_has_seen[self.similarity_matrix[new_movie][i][0]] \
                               * self.similarity_matrix[new_movie][i][1]
            # 入小根堆
            heapq.heappush(self.result, (final_score, new_movie))
            if len(self.result) > num:
                heapq.heappop(self.result)
        self.result.sort(reverse=True)
        for i in range(len(self.result)):
            self.result[i] = self.result[i][1]


CF = ItemCF('1')
CF.LoadMovieLensData()
CF.find_high_item()
CF.find_havnt_seen()
CF.cal_result()
for item in CF.result:
    print(item)
