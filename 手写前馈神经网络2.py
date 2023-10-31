import numpy as np
from scipy.stats import multivariate_normal
import random
import matplotlib.pyplot as plt
import warnings

warnings.filterwarnings("ignore")
plt.rcParams['font.sans-serif'] = ['Microsoft Yahei']


def MSE_Loss(y_true, y_pred):
    return ((y_true - y_pred) ** 2).mean()


class Net:
    def __init__(self, mode="sigmoid",
                 learning_rate=0.005, sigmoid_para=2,
                 softrelu_para=0.12, weight_para=0.6,
                 leakyrelu_para=0.02):
        self.parameters_w = []  # 储存权重
        self.parameters_b = []  # 储存偏置
        self.layer_size = 1  # 网络层数，三层神经元就有两层权重，三层网络
        self.neuron = [[0]]  # 各层神经元的值
        self.learning_rate = learning_rate
        self.deriv_layer = [[0]]  # 每层神经元导数值
        self.delta = []
        self.result = 0
        self.mode = mode
        self.sigmoid_para = sigmoid_para
        self.softrelu_para = softrelu_para
        self.init_weight = weight_para
        self.leakyrelu_para = leakyrelu_para

    def init_layer(self, numi_1, numi):  # 前后两层神经元个数
        # weighti = np.random.rand(numi, numi_1)
        weighti = np.random.normal(size=(numi, numi_1)) * self.init_weight
        bi = np.random.normal(size=numi)
        self.parameters_w.append(weighti)
        bi = bi.reshape(bi.size, 1)
        self.parameters_b.append(bi)
        self.layer_size += 1
        self.neuron.append([0])
        self.deriv_layer.append([0])
        self.delta.append([0])

    def show_layer(self):
        for i in range(self.layer_size - 1):
            temp = self.parameters_w[i]  # 输出权重
            for j in range(temp.shape[0]):
                for k in range(temp.shape[1]):
                    print(temp[j][k], end="")
                    print(' ', end="")
                print("")
            print("-----------------")
            temp = self.parameters_b[i]  # 输出偏置
            for j in range(temp.filter_size):
                print(temp[j][0], end="")
                print(' ', end="")
            print("")
            print("")

    def show_neuron(self):  # 显示每层神经元的值
        for i in range(self.layer_size):
            for j in range(len(self.neuron[i])):
                print(self.neuron[i][j], end="")
                print(" ", end="")
            print("")

    def show_deriv(self):  # 显示每层神经元导数值
        for i in range(self.layer_size):
            for j in range(len(self.deriv_layer[i])):
                print(self.deriv_layer[i][j], end="")
                print(" ", end="")
            print("")

    def sigmoid(self, x):  # 激活函数用sigmoid
        return self.sigmoid_para / (1 + np.exp(-x))

    def soft_relu(self, x):  # 激活函数用软relu函数
        return np.sqrt((x ** 2) / 4 + self.softrelu_para) + x / 2

    def leakyrelu(self, x):  # 激活函数用leakyrelu
        return ((1 + self.leakyrelu_para) * x + np.sqrt(
            (1 - self.leakyrelu_para) ** 2 * x * x + self.softrelu_para)) / 2

    def deriv_soft_relu(self, x):
        return (x ** 2 - self.softrelu_para) / (2 * x * x + 2 * self.softrelu_para) + 0.5

    def deriv_sigmoid(self, x):
        return x * (1 - x / self.sigmoid_para)

    def deriv_leakyrelu(self, x):
        return ((1 - self.leakyrelu_para) ** 2) * x / np.sqrt(
            (1 - self.leakyrelu_para) ** 2 * x * x + self.softrelu_para) / 2 \
               + 1 + self.leakyrelu_para

    def forward(self, x):  # 前向计算
        temp = np.array(x)
        if self.parameters_w[0].shape[1] == 1:  # 单输入
            temp = np.array([x])
        self.neuron[0] = temp
        temp = temp.reshape(temp.size, 1)
        for i in range(1, self.layer_size - 1):
            temp = np.dot(self.parameters_w[i - 1], temp) + self.parameters_b[i - 1]
            temp = self.sigmoid(temp)
            self.neuron[i] = temp
            self.deriv_layer[i] = self.deriv_sigmoid(temp)
        temp = np.dot(self.parameters_w[self.layer_size - 2], temp) \
               + self.parameters_b[self.layer_size - 2]
        if self.mode == "sigmoid":
            temp = self.sigmoid(temp)
            self.neuron[self.layer_size - 1] = temp
            self.deriv_layer[self.layer_size - 1] = self.deriv_sigmoid(temp)
        if self.mode == "soft-relu":  # 仅改变最后一层的非线性函数
            temp = self.soft_relu(temp)
            self.neuron[self.layer_size - 1] = temp
            self.deriv_layer[self.layer_size - 1] = self.deriv_soft_relu(temp)
        if self.mode == "leakyrelu":
            temp = self.leakyrelu(temp)
            self.neuron[self.layer_size - 1] = temp
            self.deriv_layer[self.layer_size - 1] = self.deriv_leakyrelu(temp)
        self.result = self.neuron[-1][0]

    def backpropagation(self, y_true):  # 反向传播
        """
        delta[i]表示最终损失函数对第i层线性计算结果求导，其中要保留第2层到最后一层结果，第0层结果不需保留
        具体在列表中，delta[0]表示第二层的delta值，delta[n-2]表示第n层delta值（对应neuron[n-1])
        最后一层delta是用最后一层导数以及输出差值直接计算得到，前面的层都是由后一层递归计算得到
        """
        self.delta[-1] = (self.neuron[-1] - np.array(y_true)) * self.deriv_layer[-1]
        if (self.layer_size >= 3):
            for i in range(self.layer_size - 3, -1, -1):
                self.delta[i] = self.deriv_layer[i + 1] * np.dot(
                    self.parameters_w[i + 1].T, self.delta[i + 1])
        # 接下来进行参数更新
        for i in range(self.layer_size - 1):
            self.parameters_w[i] -= self.learning_rate * (np.dot(
                self.delta[i].reshape(self.delta[i].filter_size, 1),
                self.neuron[i].reshape(1, self.neuron[i].size)))
            self.parameters_b[i] -= self.learning_rate * self.delta[i]

"""
net = Net("leakyrelu")
net.init_layer(3, 5)
net.init_layer(5, 10)
net.init_layer(10, 1)

# 构造数据集

mask_convariance_maxtix = [[5, 0, 0],
                           [0, 5, 0],
                           [0, 0, 5]]
distance = 2
center_negative = np.array([-distance, -distance, -distance])
train_datasize = 50
data_netative = multivariate_normal.rvs(mean=center_negative, cov=mask_convariance_maxtix, size=train_datasize)
# data = preprocessing(data)
y_negative = np.zeros(train_datasize)
data_netative = np.insert(data_netative, 3, y_negative, axis=1)
data = data_netative
center_positive = np.array([distance, distance, distance])
data_positive = multivariate_normal.rvs(mean=center_positive, cov=mask_convariance_maxtix, size=train_datasize)
y_positive = np.ones(train_datasize)
data_positive = np.insert(data_positive, 3, y_positive, axis=1)
data = np.insert(data_netative, train_datasize, data_positive, axis=0)
np.random.shuffle(data)
# print(data)
label = data[:, -1]
data = data[:, :3]

generation = np.array([])
final_loss = np.array([])
last_loss = 100
for i in range(50):
    temp = []
    for j in range(train_datasize * 2):
        net.forward(data[j])
        # print(label[j])
        net.backpropagation(label[j])
        temp.append(net.result[0])
    temp = np.array(temp)
    loss = MSE_Loss(label, temp)
    if abs(loss - last_loss) <= 0.001:
        break
    last_loss = loss
    final_loss = np.append(final_loss, loss)
    generation = np.append(generation, i)
    if i % 5 == 0:
        print("Epoch %d loss: %.6f" % (i, loss))

plt.plot(generation, final_loss)
plt.title("损失函数变化")
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.show()
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(data_netative[:, 0], data_netative[:, 1], data_netative[:, 2], c='r')
ax.scatter(data_positive[:, 0], data_positive[:, 1], data_positive[:, 2], c='g')
ax.set_xlabel('X axis')
ax.set_ylabel('Y axis')
ax.set_zlabel('Z axis')
plt.title("样本点分布")
plt.show()

# 测试阶段
test_datasize = 10
data_test_negative = multivariate_normal.rvs(
    mean=center_negative, cov=mask_convariance_maxtix, size=test_datasize)
y_test_negative = np.zeros(test_datasize)
data_test_negative = np.insert(data_test_negative, 3, y_test_negative, axis=1)
test_data = data_test_negative

data_test_positive = multivariate_normal.rvs(
    mean=center_positive, cov=mask_convariance_maxtix, size=test_datasize)
y_test_positive = np.ones(test_datasize)
data_test_positive = np.insert(data_test_positive, 3, y_test_positive, axis=1)
test_data = np.insert(test_data, test_datasize, data_test_positive, axis=0)
test_result = []
y_test = test_data[:, -1]
test_data = test_data[:, :3]
test_datasize = 2 * test_datasize
for i in range(test_datasize):
    net.forward(test_data[i])
    test_result.append(1 if net.result[0] >= 0.5 else 0)
test_result = np.array(test_result)
accuracy = 0
for i in range(test_datasize):
    if test_result[i] == y_test[i]:
        accuracy += 1
accuracy /= test_datasize
print(test_result)
print("accuracy:", accuracy)
"""
net = Net("leakyrelu")
net.init_layer(1, 20)
net.init_layer(20, 25)
net.init_layer(25, 1)
# 拟合三角函数
train_size = 400
x_trian = 10 * np.random.random(train_size)
y_train = np.sin(x_trian)

# 接下来进行网络训练
epoch = 250
final_loss = np.array([])
generation = np.array([])
last_loss = 100
for i in range(epoch):
    temp = []
    for j in range(train_size):
        net.forward(x_trian[j])
        net.backpropagation(y_train[j])
        temp.append(net.result[0])
    temp = np.array(temp)
    loss = MSE_Loss(y_train, temp)
    final_loss = np.append(final_loss, loss)
    generation = np.append(generation, i)
    if loss - last_loss >= 0.02:
        break
    last_loss = loss
    if i % 5 == 0:
        print("Epoch %d loss: %.6f" % (i, loss))
plt.plot(generation, final_loss)
plt.title("损失函数变化")
plt.xlabel("epoch")
plt.ylabel("Loss")
plt.show()
test_size = 100
X_test = 10 * np.random.random(test_size)
Y_test = np.array([])
for i in range(test_size):
    net.forward(X_test[i])
    Y_test = np.append(Y_test, net.result[0])
plt.scatter(X_test, Y_test, c='g')
plt.scatter(x_trian, y_train, c='r', marker='.')
plt.legend(labels = ["拟合曲线","实际曲线"])
plt.title('拟合三角函数')
plt.ylabel('Y axis')
plt.xlabel('X axis')
plt.show()

