#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2021/2/3 15:41
# Author: Hou hailun

# Course 2 - 改善深层神经网络 - 第一周作业(1&2&3) - 初始化、正则化、梯度校验

# 1. 初始化参数：
# 	1.1：使用0来初始化参数。
# 	1.2：使用随机数来初始化参数。
# 	1.3：使用抑梯度异常初始化参数（参见视频中的梯度消失和梯度爆炸）。
# 2. 正则化模型：
# 	2.1：使用二范数对二分类模型正则化，尝试避免过拟合。
# 	2.2：使用随机删除节点的方法精简模型，同样是为了尝试避免过拟合。
# 3. 梯度校验  ：对模型使用梯度校验，检测它是否在梯度下降的过程中出现误差过大的情况。


import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sklearn.datasets
from utils import init_utils   #第一部分，初始化
from utils import reg_utils    #第二部分，正则化
from utils import gc_utils     #第三部分，梯度校验

# %matplotlib inline #如果你使用的是Jupyter Notebook，请取消注释。
plt.rcParams['figure.figsize'] = (7.0, 4.0)  # set default size of plots
plt.rcParams['image.interpolation'] = 'nearest'
plt.rcParams['image.cmap'] = 'gray'


class Model_param:
    def __init__(self):
        # 读取并绘制数据
        self.train_X, self.train_Y, self.test_X, self.test_Y = init_utils.load_dataset(is_plot=False)

    def initialize_parameters_zeros(self, layers_dims):
        """
        初始化参数
        :param layers_dims: 列表，模型的层数和对应每一层的节点的数量
        :return:
        """
        parameters = {}

        L = len(layers_dims)  # 网络的层数
        for l in range(1, L):
            parameters['W' + str(l)] = np.zeros(shape=(layers_dims[l], layers_dims[l-1]))
            parameters['b' + str(l)] = np.zeros(shape=(layers_dims[l], 1))

            # 使用断言确保我的数据格式是正确的
            assert (parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
            assert (parameters["b" + str(l)].shape == (layers_dims[l], 1))

        return parameters

    def initialize_parameters_random(self, layers_dims):
        """
        随机初始化
        :param layers_dims:
        :return:
        """
        parameters = {}

        L = len(layers_dims)  # 网络的层数
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * 10
            parameters['b' + str(l)] = np.zeros(shape=(layers_dims[l], 1))

            # 使用断言确保我的数据格式是正确的
            assert (parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
            assert (parameters["b" + str(l)].shape == (layers_dims[l], 1))

        return parameters

    def initialize_parameters_he(self, layers_dims):
        """
        抑制度异常初始化
        :param layers_dims:
        :return:
        """
        parameters = {}

        L = len(layers_dims)  # 网络的层数
        for l in range(1, L):
            parameters['W' + str(l)] = np.random.randn(layers_dims[l], layers_dims[l - 1]) * np.sqrt(2 / layers_dims[l-1])
            parameters['b' + str(l)] = np.zeros(shape=(layers_dims[l], 1))

            # 使用断言确保我的数据格式是正确的
            assert (parameters["W" + str(l)].shape == (layers_dims[l], layers_dims[l - 1]))
            assert (parameters["b" + str(l)].shape == (layers_dims[l], 1))

        return parameters

    def model(self, X, Y, learning_rate=0.01, num_iterations=15000, print_cost=True, initialization="he", is_polt=True):
        """
        实现一个三层的神经网络：LINEAR ->RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID

        参数：
            X - 输入的数据，维度为(2, 要训练/测试的数量)
            Y - 标签，【0 | 1】，维度为(1，对应的是输入的数据的标签)
            learning_rate - 学习速率
            num_iterations - 迭代的次数
            print_cost - 是否打印成本值，每迭代1000次打印一次
            initialization - 字符串类型，初始化的类型【"zeros" | "random" | "he"】
            is_polt - 是否绘制梯度下降的曲线图
        返回
            parameters - 学习后的参数
        """
        grads = {}
        costs = []
        m = X.shape[1]
        layers_dims = [X.shape[0], 10, 5, 1]  # 三层网络结构

        # 选择初始化参数的类型
        if initialization == 'zeros':
            parameters = self.initialize_parameters_zeros(layers_dims)
        elif initialization == 'random':
            parameters = self.initialize_parameters_random(layers_dims)
        elif initialization == 'he':
            parameters = self.initialize_parameters_he(layers_dims)
        else:
            print('错误的初始化参数')
            exit()

        # 开始学习
        for i in range(num_iterations):
            # 前向传播
            a3, cache = init_utils.forward_propagation(X, parameters)

            # 计算损失
            cost = init_utils.compute_loss(a3, Y)

            # 反向传播
            grads = init_utils.backward_propagation(X, Y, cache)

            # 更新参数
            parameters = init_utils.update_parameters(parameters, grads, learning_rate)

            # 记录损失
            if i % 1000 == 0:
                costs.append(cost)
                if print_cost:
                    print("第" + str(i) + "次迭代，成本值为：" + str(cost))

        # 学习完毕，绘制成本曲线
        if is_polt:
            plt.plot(costs)
            plt.ylabel('cost')
            plt.xlabel('iterations (per hundreds)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

        return parameters


class Model_Normal:
    # 带正则化的模型
    # 不使用正则化
    # 使用正则化
    #   2.1 使用L2正则化
    #   2.2 使用随机节点删除
    def __init__(self):
        self.train_X, self.train_Y, self.test_X, self.test_Y = reg_utils.load_2D_dataset(is_plot=False)
        # 每一个点代表球落下的可能的位置，蓝色代表己方的球员会抢到球，红色代表对手的球员会抢到球，
        # 我们要做的就是使用模型来画出一条线，来找到适合我方球员能抢到球的位置

    def compute_cost_with_regularization(self, A3, Y, parameters, lambd):
        """
        实现公式2的L2正则化计算成本

        参数：
            A3 - 正向传播的输出结果，维度为（输出节点数量，训练/测试的数量）
            Y - 标签向量，与数据一一对应，维度为(输出节点数量，训练/测试的数量)
            parameters - 包含模型学习后的参数的字典
        返回：
            cost - 使用公式2计算出来的正则化损失的值
        """
        m = Y.shape[1]
        W1 = parameters['W1']
        W2 = parameters['W2']
        W3 = parameters['W3']

        cross_entropy_cost = reg_utils.compute_cost(A3, Y)

        L2_regularization_cost = lambd * (np.sum(np.squeeze(W1)) + np.sum(np.squeeze(W2)) + np.sum(np.squeeze(W3))) / (2 * m)

        cost = cross_entropy_cost + L2_regularization_cost

        return cost

    def backward_propagation_with_regularization(self, X, Y, cache, lambd):
        """
        实现我们添加了L2正则化的模型的后向传播。

        参数：
            X - 输入数据集，维度为（输入节点数量，数据集里面的数量）
            Y - 标签，维度为（输出节点数量，数据集里面的数量）
            cache - 来自forward_propagation（）的cache输出
            lambda - regularization超参数，实数

        返回：
            gradients - 一个包含了每个参数、激活值和预激活值变量的梯度的字典
        """
        m = X.shape[1]

        (Z1, A1, W1, b1, Z2, A2, W2, b2, Z3, A3, W3, b3) = cache

        dZ3 = A3 - Y

        dW3 = (1 / m) * np.dot(dZ3, A2.T) + ((lambd * W3) / m)
        db3 = (1 / m) * np.sum(dZ3, axis=1, keepdims=True)

        dA2 = np.dot(W3.T, dZ3)
        dZ2 = np.multiply(dA2, np.int64(A2 > 0))
        dW2 = (1 / m) * np.dot(dZ2, A1.T) + ((lambd * W2) / m)
        db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(W2.T, dZ2)
        dZ1 = np.multiply(dA1, np.int64(A1 > 0))
        dW1 = (1 / m) * np.dot(dZ1, X.T) + ((lambd * W1) / m)
        db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

        gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                     "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                     "dZ1": dZ1, "dW1": dW1, "db1": db1}

        return gradients

    def forward_propagation_with_dropout(X, parameters, keep_prob=0.5):
        """
        实现具有随机舍弃节点的前向传播。
        LINEAR -> RELU + DROPOUT -> LINEAR -> RELU + DROPOUT -> LINEAR -> SIGMOID.

        参数：
            X  - 输入数据集，维度为（2，示例数）
            parameters - 包含参数“W1”，“b1”，“W2”，“b2”，“W3”，“b3”的python字典：
                W1  - 权重矩阵，维度为（20,2）
                b1  - 偏向量，维度为（20,1）
                W2  - 权重矩阵，维度为（3,20）
                b2  - 偏向量，维度为（3,1）
                W3  - 权重矩阵，维度为（1,3）
                b3  - 偏向量，维度为（1,1）
            keep_prob  - 随机删除的概率，实数
        返回：
            A3  - 最后的激活值，维度为（1,1），正向传播的输出
            cache - 存储了一些用于计算反向传播的数值的元组
        """
        np.random.seed(3)

        W1 = parameters['W1']
        W2 = parameters['W2']
        W3 = parameters['W3']
        b1 = parameters['b1']
        b2 = parameters['b2']
        b3 = parameters['b3']

        # LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID
        Z1 = np.dot(W1, X) + b1
        A1 = reg_utils.relu(Z1)

        D1 = np.random.rand(A1.shape[0], A1.shape[1])  # 步骤1: 初始化矩阵D1
        D1 = D1 < keep_prob                             # 步骤2: 将D1的值转换为0或1（使用keep_prob作为阈值）
        A1 = A1 * D1                                    # 步骤3：舍弃A1的一些节点（将它的值变为0或False）
        A1 = A1 / keep_prob                             # 步骤4：缩放未舍弃的节点(不为0)的值

        Z2 = np.dot(W2, A1) + b1
        A2 = reg_utils.relu(Z2)

        D2 = np.random.rand(A2.shape[0], A2.shape[1])
        D2 = D2 < keep_prob
        A2 = A2 * D2
        A2 = A2 / keep_prob

        Z3 = np.dot(W3, A2) + b3
        A3 = reg_utils.sigmoid(Z3)

        cache = (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3)

        return A3, cache

    def backward_propagation_with_dropout(self, X, Y, cache, keep_prob):
        """
        实现我们随机删除的模型的后向传播。
        参数：
            X  - 输入数据集，维度为（2，示例数）
            Y  - 标签，维度为（输出节点数量，示例数量）
            cache - 来自forward_propagation_with_dropout（）的cache输出
            keep_prob  - 随机删除的概率，实数

        返回：
            gradients - 一个关于每个参数、激活值和预激活变量的梯度值的字典
        """
        m = X.shape[1]
        (Z1, D1, A1, W1, b1, Z2, D2, A2, W2, b2, Z3, A3, W3, b3) = cache

        dZ3 = A3 - Y
        dW3 = (1 / m) * np.dot(dZ3, A2.T)
        db3 = 1. / m * np.sum(dZ3, axis=1, keepdims=True)
        dA2 = np.dot(W3.T, dZ3)

        dA2 = dA2 * D2  # 步骤1：使用正向传播期间相同的节点，舍弃那些关闭的节点（因为任何数乘以0或者False都为0或者False）
        dA2 = dA2 / keep_prob  # 步骤2：缩放未舍弃的节点(不为0)的值

        dZ2 = np.multiply(dA2, np.int64(A2 > 0))
        dW2 = 1. / m * np.dot(dZ2, A1.T)
        db2 = 1. / m * np.sum(dZ2, axis=1, keepdims=True)

        dA1 = np.dot(W2.T, dZ2)

        dA1 = dA1 * D1  # 步骤1：使用正向传播期间相同的节点，舍弃那些关闭的节点（因为任何数乘以0或者False都为0或者False）
        dA1 = dA1 / keep_prob  # 步骤2：缩放未舍弃的节点(不为0)的值

        dZ1 = np.multiply(dA1, np.int64(A1 > 0))
        dW1 = 1. / m * np.dot(dZ1, X.T)
        db1 = 1. / m * np.sum(dZ1, axis=1, keepdims=True)

        gradients = {"dZ3": dZ3, "dW3": dW3, "db3": db3, "dA2": dA2,
                     "dZ2": dZ2, "dW2": dW2, "db2": db2, "dA1": dA1,
                     "dZ1": dZ1, "dW1": dW1, "db1": db1}

        return gradients

    def model(self, X, Y, learning_rate=0.3, num_iterations=30000, print_cost=True,
              is_plot=True, lambd=0, keep_prob=1):
        """
        实现一个三层的神经网络：LINEAR ->RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID

        参数：
            X - 输入的数据，维度为(2, 要训练/测试的数量)
            Y - 标签，【0(蓝色) | 1(红色)】，维度为(1，对应的是输入的数据的标签)
            learning_rate - 学习速率
            num_iterations - 迭代的次数
            print_cost - 是否打印成本值，每迭代10000次打印一次，但是每1000次记录一个成本值
            is_polt - 是否绘制梯度下降的曲线图
            lambd - 正则化的超参数，实数
            keep_prob - 随机删除节点的概率
        返回
            parameters - 学习后的参数
        """
        grad = {}
        costs = []
        m = X.shape[1]
        layers_dims = [X.shape[0], 20, 3, 1]

        # 初始化参数
        parameters = reg_utils.initialize_parameters(layers_dims)

        # 开始学习
        for i in range(num_iterations):
            # 前向传播
            # 是否随机删除节点
            if keep_prob == 1:  # 不随机删除节点
                a3, cache = reg_utils.forward_propagation(X, parameters)
            elif keep_prob < 1:  # 随机删除节点
                a3, cache = self.forward_propagation_with_dropout(X, parameters, keep_prob)
            else:
                print("keep_prob参数错误！程序退出。")
                exit()

            # 计算损失
            # 是否使用二范数
            if lambd == 0:  # 不使用l2正则化
                cost = reg_utils.compute_cost(a3, Y)
            else:           # 使用L2正则化
                cost = self.compute_cost_with_regularization(a3, Y, parameters, lambd)

            # 反向传播
            # 可以同时使用L2正则化和随机删除节点，但是本次实验不同时使用。
            assert (lambd == 0 or keep_prob == 1)

            # 两个参数的使用情况
            if lambd == 0 and keep_prob == 1:
                # 不使用L2正则化和不使用随机删除节点
                grads = reg_utils.backward_propagation(X, Y, cache)
            elif lambd != 0:
                # 使用L2正则化，不使用随机删除节点
                grads = self.backward_propagation_with_regularization(X, Y, cache, lambd)
            elif keep_prob < 1:
                # 使用随机删除节点，不使用L2正则化
                grads = self.backward_propagation_with_dropout(X, Y, cache, keep_prob)

            # 更新参数
            parameters = reg_utils.update_parameters(parameters, grads, learning_rate)

            # 记录并打印成本
            if i % 1000 == 0:
                # 记录成本
                costs.append(cost)
                if print_cost and i % 10000 == 0:
                    # 打印成本
                    print("第" + str(i) + "次迭代，成本值为：" + str(cost))

        # 是否绘制成本曲线图
        if is_plot:
            plt.plot(costs)
            plt.ylabel('cost')
            plt.xlabel('iterations (x1,000)')
            plt.title("Learning rate =" + str(learning_rate))
            plt.show()

            # 返回学习后的参数
            return parameters


def model_normal_test():
    model = Model_Normal()

    # 测试一：不带正则化
    # parameters = model.model(model.train_X, model.train_Y, is_plot=True)
    #
    # print("训练集:")
    # predictions_train = reg_utils.predict(model.train_X, model.train_Y, parameters)
    # print("测试集:")
    # predictions_test = reg_utils.predict(model.test_X, model.test_Y, parameters)
    #
    # plt.title("Model without regularization")
    # axes = plt.gca()
    # axes.set_xlim([-0.75, 0.40])
    # axes.set_ylim([-0.75, 0.65])
    # reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), model.train_X, np.squeeze(model.train_Y))
    # # 在无正则化时，分割曲线有了明显的过拟合特性

    # 测试二：使用L2正则化
    # parameters = model.model(model.train_X, model.train_Y, lambd=0.7, is_plot=True)
    # print("使用正则化，训练集:")
    # predictions_train = reg_utils.predict(model.train_X, model.train_Y, parameters)
    # print("使用正则化，测试集:")
    # predictions_test = reg_utils.predict(model.test_X, model.test_Y, parameters)
    #
    # plt.title("Model with L2-regularization")
    # axes = plt.gca()
    # axes.set_xlim([-0.75, 0.40])
    # axes.set_ylim([-0.75, 0.65])
    # reg_utils.plot_decision_boundary(lambda x: reg_utils.predict_dec(parameters, x.T), model.train_X,  np.squeeze(model.train_Y))



def model_param_test():
    model = Model_param()

    parameters = model.model(model.train_X, model.train_Y, initialization='he', is_polt=True)

    print(parameters)

    print("训练集:")
    predictions_train = init_utils.predict(model.train_X, model.train_Y, parameters)
    print("测试集:")
    predictions_test = init_utils.predict(model.test_X, model.test_Y, parameters)

    # 预测和决策边界的细节
    print("predictions_train = " + str(predictions_train))
    print("predictions_test = " + str(predictions_test))

    plt.title("Model with Zeros initialization")
    axes = plt.gca()
    axes.set_xlim([-1.5, 1.5])
    axes.set_ylim([-1.5, 1.5])
    init_utils.plot_decision_boundary(lambda x: init_utils.predict_dec(parameters, x.T), model.train_X, model.train_Y)

    # 结论：
    # 参数0初始化：分类失败，该模型预测每个都为0。通常来说，零初始化都会导致神经网络无法打破对称性，最终导致的结果就是无论网络有多少层，最终只能得到和Logistic函数相同的效果
    # 不同的初始化方法可能导致性能最终不同
    # 随机初始化有助于打破对称，使得不同隐藏层的单元可以学习到不同的参数。
    # 初始化时，初始值不宜过大。
    # He初始化搭配ReLU激活函数常常可以得到不错的效果


if __name__ == "__main__":
    # model_param_test()

    model_normal_test()


