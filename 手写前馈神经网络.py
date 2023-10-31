# 三层卷积神经网络
import numpy as np
import matplotlib.pyplot as plt
import random


class Net:
    def __init__(self):
        self.parameters_w = []  # 储存权重
        self.parameters_b = []  # 储存偏置
        self.layer_size = 0  # 权重层数，三层神经元就有两层权重
        self.init_layer(1, 10)
        self.init_layer(10, 20)
        self.init_layer(20, 10)
        self.init_layer(10, 5)
        self.init_layer(5, 1)
        self.neuron = [0] * self.layer_size  # 各层神经元的值
        self.learning_rate = 0.008
        self.relu_layer = [0] * self.layer_size  # 看每一层神经元对应relu导数是1还是0

    def init_layer(self, numi_1, numi):  # 前后两层神经元个数
        # weighti = np.random.rand(numi, numi_1)
        weighti = np.random.normal(size=(numi, numi_1)) * 0.1
        bi = np.random.normal(size=numi) * 0
        self.parameters_w.append(weighti)
        self.parameters_b.append(bi)
        self.layer_size += 1

    def show_layer(self):
        for i in range(self.layer_size):
            temp = self.parameters_w[i]  # 输出权重
            for j in range(temp.shape[0]):
                for k in range(temp.shape[1]):
                    print(temp[j][k], end="")
                    print(' ', end="")
                print("")
            print("-----------------")
            temp = self.parameters_b[i]  # 输出偏置
            for j in range(temp.filter_size):
                print(temp[j], end="")
                print(' ', end="")
            print("")
            print("")

    def show_neuron(self):
        for i in range(self.layer_size):
            for j in range(len(self.neuron[i])):
                print(self.neuron[i][j], end="")
                print(" ", end="")
            print("")
            print("-----------------")

    def relu(self, x):  # 中间层激活函数用relu
        return np.maximum(0.01 * x, x)

    def sigmoid(self, x):  # 最后一层激活函数用softmax
        return 1 / (1 + np.exp(-x))

    def forward(self, x):
        # print(self.parameters_w[0].shape[1])
        if self.parameters_w[0].shape[1] == 1:
            temp = np.array([x])  # 单输入
        else:
            temp = np.array(x)
        self.neuron[0] = temp
        for i in range(self.layer_size - 1):
            temp = np.dot(self.parameters_w[i], temp) + self.parameters_b[i]
            temp = self.relu(temp)
            temp_relu = np.array([1 if j >= 0 else 0.01 for j in temp])
            self.relu_layer[i + 1] = temp_relu
            self.neuron[i + 1] = temp
        temp = np.dot(self.parameters_w[self.layer_size - 1], temp) + self.parameters_b[self.layer_size - 1]
        temp = self.relu(temp)
        return temp

    def backward(self, y_temp, y_target):
        """
        :param y_temp: 当前计算出的目标值
        :param y_target: 期望的目标值
        :return: 更新整套网络参数
        """
        # 首先存储整套网络每个神经元误差
        delta = []
        temp_number = 0  # 当前计算到了第几层
        temp_forward = y_temp - y_target  # 最前面一层输出的误差
        delta.append(temp_forward)  # 注意delta第一个元素是最后输出层误差，最后一个元素是第一层隐藏层误差
        for i in range(self.layer_size - 1, 0, -1):
            temp_forward = np.dot(delta[temp_number], self.parameters_w[i]) * self.relu_layer[i]
            temp_number += 1
            delta.append(temp_forward)
        return delta

    def train(self, train_data, train_target):
        """
        :param train_data: 训练数据集
        :param train_target: 训练标签
        :return: 平均delta
        """
        data_size = len(train_data)
        # print(data_size)
        delta_all = []
        loss = 0
        for i in range(data_size):
            temp = self.forward(train_data[i])
            delta = self.backward(temp, train_target[i])
            loss += (temp - train_target[i]) * (temp - train_target[i]) / 2
            delta_all.append(delta)
        delta = delta_all[0]
        loss /= data_size

        for j in range(len(delta_all[0])):
            for i in range(1, data_size):
                delta[j] += delta_all[i][j]
            delta[j] /= data_size
        """
        for i in range(len(delta)):
            for j in range(len(delta[i])):
                print(delta[i][j],end="")
                print(" ",end="")
            print("")
        """
        self.gradient(delta)
        return loss

    def gradient(self, delta):
        # 接下来计算梯度并更新
        for i in range(self.layer_size):
            temp1 = delta[self.layer_size - 1 - i].reshape(delta[self.layer_size - 1 - i].shape[0], 1)
            # print(temp1.shape,(self.neuron[i].shape))
            # print(type(temp1),type(self.neuron[i]))
            temp2 = self.neuron[i].reshape(1, (self.neuron[i]).size)
            # print(temp1.size, temp2.size)
            delta_w = np.dot(temp1, temp2)
            # delta_w = delta_w.reshape(self.parameters_w[i].shape[0],self.parameters_w[i].shape[1])
            # print(delta_w.shape)
            # print(self.parameters_w[i].shape)
            # print(" ")
            self.parameters_w[i] -= (self.learning_rate * delta_w)
            self.parameters_b[i] -= self.learning_rate * delta[self.layer_size - 1 - i]


mymodel = Net()
"""
input = 2
target = 10
for i in range(1000):
    temp = mymodel.forward(input)
    if i % 50 == 0:
        print(temp)
    if abs(target-temp)<=0.000001:
        break
    delta = mymodel.backward(temp, target)
    mymodel.gradient(delta)
"""
X_standard = np.linspace(0, 10, 200)
Y = np.sin(X_standard)
plt.plot(X_standard, Y)
train_size = 500
X_train = 10 * np.random.random(train_size)
Y_train = np.sin(X_train)

# 接下来进行网络训练

epoch = 20
iteration = 25
k = int(train_size / iteration)
loss_all = []
for i in range(epoch):
    loss_temp = 0
    for j in range(iteration):
        temp_x = X_train[k * j:k * (j + 1)]
        temp_x = (temp_x - temp_x.mean()) / temp_x.std()
        temp_y = Y_train[k * j:k * (j + 1)]
        loss = mymodel.train(temp_x, temp_y)
        loss_temp += loss
    loss_all.append(loss_temp / iteration)
    # print(loss_temp / iteration)

test_size = 50
X_test = 10 * np.random.random(test_size)
Y_test = np.array([])
for i in range(test_size):
    Y_test = np.append(Y_test, mymodel.forward(X_test[i]))
print(Y_test)
plt.scatter(X_test, Y_test, c='g')
plt.show()
plt.scatter(X_test, Y_test, c='r', marker='.')
plt.show()
plt.plot(range(epoch), loss_all)
plt.show()
