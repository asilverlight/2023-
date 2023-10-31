import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from math import pow, sqrt, log1p
import heapq
import random
from collections import defaultdict
from operator import itemgetter

df = pd.read_csv("winequality-red.csv", sep=';')


class Bayes:
    def __init__(self, rating=0.8):
        self.data = []  # 存储数据，其中数据按照quality排好
        self.train = []
        self.test = []
        self.train_num = 0
        self.test_num = 0
        self.rating = rating  # 抽取多少数据作为训练集
        self.character = []  # 特征种类
        self.character_kind = dict()  # 每种特征能取多少类
        """
        self.character_kind
        {'色泽': 3, '根蒂': 3, '敲声': 3, '纹理': 3, '脐部': 3, '触感': 2}
        """
        self.label_kind = dict()  # 每种标签有多少个
        """
        self.label_kind
        {1: 6, 0: 6}
        """
        self.label_prob = dict()  # 每种quality对应概率，按quality从低到高排好
        """
        {0: 0.6, 1: 0.4}
        """
        self.cond_prob = dict()  # cond_prob[i,j,k]代表在Y类别为i的情况下，X的第j个特征为第k种的概率
        """
        self.cond_prob
        {'0': 
            {'色泽': 
                {'乌黑': 1.0, '浅白': 4.0, '青绿': 1.0}, 
            '根蒂': 
                {'稍蜷': 3.0, '硬挺': 1.0, '蜷缩': 2.0}, 
            '敲声': 
                {'沉闷': 2.0, '清脆': 1.0, '浊响': 3.0}, 
            '纹理': 
                {'稍糊': 3.0, '模糊': 3.0}, 
            '脐部': 
                {'稍凹': 1.0, '平坦': 3.0, '凹陷': 2.0}, 
            '触感': 
                {'硬滑': 5.0, '软粘': 1.0}}, 
        '1': 
            {'色泽': 
                {'青绿': 3.0, '乌黑': 4.0, '浅白': 1.0}, 
            '根蒂': 
                {'蜷缩': 5.0, '稍蜷': 3.0}, 
            '敲声': 
                {'浊响': 6.0, '沉闷': 2.0}, 
            '纹理': 
                {'清晰': 7.0, '稍糊': 1.0}, 
            '脐部': 
                {'凹陷': 5.0, '稍凹': 3.0}, 
            '触感': 
                {'硬滑': 6.0, '软粘': 2.0}}}
        """
        # 其中，cond_prob为一个三维向量

    def LoadData(self):
        self.data = pd.read_csv("data_word.csv")
        self.character = pd.Index(self.data.keys())
        self.character = self.character.tolist()
        self.character.pop()
        """
        self.character
        ['色泽', '根蒂', '敲声', '纹理', '脐部', '触感']
        """
        self.data = self.data.values.tolist()
        """
        self.data
        [['青绿', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 1], 
        ['乌黑', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 1], 
        ['乌黑', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 1], 
        ['青绿', '蜷缩', '沉闷', '清晰', '凹陷', '硬滑', 1], 
        ['浅白', '蜷缩', '浊响', '清晰', '凹陷', '硬滑', 1], 
        ['青绿', '稍蜷', '浊响', '清晰', '稍凹', '软粘', 1]..."""
        for i in range(len(self.data)):
            if random.random() < self.rating:
                self.train.append(self.data[i])
                self.train_num += 1
                self.label_kind.setdefault(self.data[i][-1], 0)
                self.label_kind[self.data[i][-1]] += 1
            else:
                self.test.append(self.data[i])
                self.test_num += 1

        self.train.sort(key=lambda x: x[-1], reverse=False)
        self.train = np.array(self.train)
        for i in range(len(self.character)):
            temp_set = set(self.train[:, i])
            self.character_kind[self.character[i]] = len(temp_set)

    def cal_label_prob(self):  # 计算每种quality对应概率
        for item in self.train:
            self.label_prob.setdefault(item[-1], float(0))
            self.label_prob[item[-1]] += 1
        for item in self.label_prob.keys():
            self.label_prob[item] /= self.train_num

    def cal_cond_prob(self):  # 计算条件概率
        for train_X in self.train:
            temp1 = train_X[-1]
            self.cond_prob.setdefault(temp1, dict())
            for cha in range(len(self.character)):
                self.cond_prob[temp1].setdefault(self.character[cha], dict())
                self.cond_prob[temp1][self.character[cha]].setdefault(train_X[cha], float(0))
                self.cond_prob[temp1][self.character[cha]][train_X[cha]] += 1
        for label in self.cond_prob.keys():
            down = self.label_kind[int(label)]
            for cha1 in self.cond_prob[label].keys():
                for cha2 in self.cond_prob[label][cha1].keys():
                    self.cond_prob[label][cha1][cha2] += 1
                    self.cond_prob[label][cha1][cha2] /= (down + self.character_kind[cha1])

    def cal_result(self, test):  # 计算最终结果
        result = dict()
        for label in self.label_prob.keys():
            result.setdefault(label, self.label_prob[label])
            for i in range(len(test) - 1):
                if test[i] not in self.cond_prob[label][self.character[i]].keys():
                    result[label] /= self.character_kind[self.character[i]]
                else:
                    result[label] *= self.cond_prob[label][self.character[i]][test[i]]
        result = sorted(result.items(), key=lambda x: x[1], reverse=True)
        return int(result[0][0])

    def final_res(self):
        num = 0
        for item in self.test:
            result = self.cal_result(item)
            if result == item[-1]:
                num += 1
        return num / self.test_num


bayes = Bayes(rating=0.9)
bayes.LoadData()
bayes.cal_label_prob()
bayes.cal_cond_prob()
print(bayes.final_res())
