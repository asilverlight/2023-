import mnist
import numpy as np
from math import ceil


class Conv:
    def __init__(self, size, in_channels=1, out_channels=1, padding=0,
                 mode="valid", pooling="MaxPooling", pooling_size=2):
        """
        :param size: 卷积核尺寸
        :param in_channels: 输入通道
        :param out_channels: 输出通道
        :param padding: 填充
        :param mode: 填充模式
        :param pooling: 池化模式
        :param pooling_size: 池化尺寸
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.filter_size = size
        self.filters = (np.random.randn(out_channels, in_channels, size, size) + 1) / (size * size)
        self.padding = padding
        self.mode = mode
        self.padding_input = []  # 填充后的输入结果
        self.last_input = []  # 经过卷积操作后，未池化的结果
        self.b = np.random.randn(out_channels) / (size * size)
        self.output = []
        self.pooling = pooling
        self.pooling_size = pooling_size
        self.mytype = "conv"  # 该层为卷积层
        self.delta_pool_input = []  # 从池化层输入（卷积输出）时的delta
        self.delta_conv_input = []  # 从卷积输入的delta

    def conv_forward(self, input):
        d, h, w = input.shape
        temp_input = []
        output = []
        self.delta_conv_input = np.zeros(self.filters.shape)
        if self.mode == "valid":  # 不填充
            temp_input = input
            output = np.zeros((self.out_channels, h - self.filter_size + 1, w - self.filter_size + 1))
        if self.mode == "zero":  # 填充0
            temp_input = np.zeros((d, h + 2 * self.padding, w + 2 * self.padding))
            temp_input[:, self.padding:h + self.padding, self.padding:w + self.padding] \
                = input
            output = np.zeros((self.out_channels, h - self.filter_size + 1 + 2 * self.padding,
                               w - self.filter_size + 1 + 2 * self.padding))
        self.delta_pool_input = output
        self.padding_input = temp_input
        h = h - self.filter_size + 1 + 2 * self.padding
        w = w - self.filter_size + 1 + 2 * self.padding
        for i in range(h):
            for j in range(w):
                temp_part = self.padding_input[:, i:i + self.filter_size, j:j + self.filter_size]
                output[:, i, j] = np.sum(temp_part * self.filters[:], axis=(1, 2, 3)) + self.b[:]
        self.output = output
        self.last_input = output  # 存储池化层输入

    def pool_forward(self):
        p = self.pooling_size
        d = self.output.shape[0]
        h1 = self.output.shape[1]
        w1 = self.output.shape[2]
        h2 = ceil(h1 / self.pooling_size)
        w2 = ceil(w1 / self.pooling_size)
        output = np.zeros((d, h2, w2))
        psize = self.pooling_size
        for j in range(h2):
            for k in range(w2):
                temp_part = self.output[:, j * psize:(np.minimum((j + 1) * psize, h1)),
                            k * psize:(np.minimum((k + 1) * psize, w1))]
                if self.pooling == "MaxPooling":
                    output[:, j, k] = np.max(temp_part, axis=(1, 2))  # , axis=(1, 2)
                    # print(output[i, j, k])
                if self.pooling == "AveragePooling":
                    output[:, j, k] = np.mean(temp_part, axis=(1, 2))  # , axis=(1, 2)
        self.output = output

    def forward(self, input):
        self.conv_forward(input)
        self.pool_forward()


class FC:
    def __init__(self, in_channels, out_channels,
                 sigmoid_para=2, softrelu_para=0.12,
                 weight_para=0.6, leakyrelu_para=0.02, mode="sigmoid"):
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.output = []  # 神经元值
        self.deriv_layer = []  # 神经元导数值
        self.delta = []  # delta值
        self.weights = np.random.randn(out_channels, in_channels)
        self.bias = np.random.randn(out_channels)
        self.sigmoid_para = sigmoid_para
        self.softrelu_para = softrelu_para
        self.init_weight = weight_para
        self.leakyrelu_para = leakyrelu_para
        self.mode = mode
        self.mytype = "FC"

    def sigmoid(self, x):  # 激活函数用sigmoid
        y = x.copy()  # 对sigmoid函数优化，避免出现极大的数据溢出
        y[x >= 0] = 1.0 / (1 + np.exp(-x[x >= 0]))
        y[x < 0] = np.exp(x[x < 0]) / (1 + np.exp(x[x < 0]))
        return y

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

    def forward(self, input):
        input = input.flatten()
        self.output = self.sigmoid(np.dot(self.weights, input) + self.bias)
        self.deriv_layer = self.deriv_sigmoid(self.output)


class Net:
    def __init__(self):
        self.layers = []
        self.size = 0
        self.output = []  # 最后一层输出
        self.para_number = 0
        self.learning_rate = 0.01

    def add_layer(self, conv):
        self.layers.append(conv)
        self.size += 1
        if conv.mytype == "conv":
            self.para_number += conv.in_channels * conv.out_channels * \
                                conv.filter_size * conv.filter_size
        if conv.mytype == "FC":
            self.para_number += conv.in_channels * conv.out_channels + conv.out_channels

    def forward(self, input):
        temp = input
        for i in range(self.size):
            self.layers[i].forward(temp)
            temp = self.layers[i].output
        self.output = temp

    def show_layer(self):
        for i in range(self.size):
            if self.layers[i].mytype == "conv":
                print("conv:")
                print(self.layers[i].filters)
                print("pooling:")
                print(self.layers[i].pooling)
            if self.layers[i].mytype == "FC":
                print("FC:")
                print(self.layers[i].weights)
                print(self.layers[i].bias)
            print("-------------")

    def cross_entropy(self, label):
        loss = -np.log(self.output[label])
        acc = 1 if np.argmax(self.output) == label else 0
        return loss, acc

    def backpropagation(self, label):
        self.layers[-1].delta = np.zeros(self.layers[-1].out_channels)
        self.layers[-1].delta[label] = -1 / self.output[label]  # 最后一层按照交叉熵损失函数计算导数
        temp_size = self.size - 2  # 从倒数第二层往前反向传播
        self.layers[-1].weights -= self.learning_rate * np.dot(
            self.layers[-1].delta.reshape(self.layers[-1].delta.size, 1),
            self.layers[-2].output.reshape(1, self.layers[-2].output.size))
        self.layers[-1].bias -= self.learning_rate * self.layers[-1].delta  # 最后一层全连接层参数更新

        while (self.layers[temp_size].mytype == "FC"):  # 是全连接层，按照全连接层传递公式计算
            self.layers[temp_size].delta = np.dot(self.layers[temp_size + 1].weights.T, (
                    self.layers[temp_size + 1].delta * self.layers[temp_size + 1].deriv_layer))
            self.layers[temp_size].weights -= self.learning_rate * np.dot(
                self.layers[temp_size].delta.reshape(
                    self.layers[temp_size].delta.size, 1),
                self.layers[temp_size - 1].output.reshape(
                    1, self.layers[temp_size - 1].output.size))
            self.layers[temp_size].bias -= self.learning_rate * \
                                           self.layers[temp_size].delta
            # if self.layers[temp_size - 1].mytype == "FC":
            temp_size -= 1

        pool_temp_delta = np.dot(self.layers[temp_size + 1].weights.T, (
                self.layers[temp_size + 1].delta * self.layers[temp_size + 1].deriv_layer))
        pool_temp_delta = np.reshape(pool_temp_delta, (pool_temp_delta.size, 1, 1))
        # print(pool_temp_delta.shape)
        # 最后一层池化层的delta

        # 池化层反向传播，只实现最大池化
        while temp_size >= 0:  # 没到最底层
            if self.layers[temp_size + 1].mytype != "FC":  # 该层上一层仍是卷积层，非池化层
                pool_temp_delta = self.layers[temp_size + 1].delta_conv_input
            d = self.layers[temp_size].delta_pool_input.shape[0]
            h1 = self.layers[temp_size].delta_pool_input.shape[1]
            w1 = self.layers[temp_size].delta_pool_input.shape[2]
            h2 = self.layers[temp_size].output.shape[1]
            w2 = self.layers[temp_size].output.shape[2]
            psize = self.layers[temp_size].pooling_size
            for i in range(h2):
                for j in range(w2):
                    temp_part = self.layers[temp_size].last_input[:,
                                i * psize:(np.minimum((i + 1) * psize, h1)),
                                j * psize:(np.minimum((j + 1) * psize, w1))]  # 小块
                    temp_amax = np.amax(temp_part, axis=(1, 2))
                    for k in range(d):
                        idx = np.where(temp_part[k, :, :] == temp_amax[k])
                        id_h = idx[0][0]
                        id_w = idx[1][0]  # 找到行列
                        self.layers[temp_size].delta_pool_input[k, i * psize + id_h, j * psize + id_w] \
                            = pool_temp_delta[k, i, j]

            temp_size -= 1


test_images = mnist.test_images()[0]
test_labels = mnist.test_labels()[0]
test_images = np.reshape(test_images, (1, test_images.shape[0], test_images.shape[1]))
# print(test_images.shape)
conv1 = Conv(size=3, in_channels=1, out_channels=3, mode="valid")
conv2 = Conv(size=5, in_channels=3, out_channels=3, mode="valid")
conv3 = Conv(size=3, in_channels=3, out_channels=3, padding=2, mode="zero")
conv4 = Conv(size=3, in_channels=3, out_channels=3, mode="valid")
fc1 = FC(3, 5)
fc2 = FC(5, 5)
fc3 = FC(5, 5)
fc4 = FC(5, 10)
net = Net()
net.add_layer(conv1)
net.add_layer(conv2)
net.add_layer(conv3)
net.add_layer(conv4)
net.add_layer(fc1)
net.add_layer(fc2)
net.add_layer(fc3)
net.add_layer(fc4)
print(net.para_number)
net.forward(test_images)
net.backpropagation(test_labels)
"""
final_loss = 0
accuracy = 0
for i, (image, label) in enumerate(zip(test_images, test_labels)):
    image = np.reshape(image, (1, image.shape[0], image.shape[1]))
    net.forward(image)
    loss, acc = net.cross_entropy(label)
    final_loss += loss
    accuracy += acc
    if i % 100 == 99:
        print(
            '[Step %d] Past 100 steps: Average Loss %.3f | Accuracy: %d%%' %
            (i + 1, final_loss / 100, accuracy)
        )
        final_loss = 0
        accuracy = 0
"""
