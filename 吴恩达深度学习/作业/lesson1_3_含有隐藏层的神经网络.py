#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2021/1/29 16:29
# Author: Hou hailun

import numpy as np
import matplotlib.pyplot as plt
from testCases import *
import sklearn
import sklearn.datasets
import sklearn.linear_model
from planar_utils import plot_decision_boundary, sigmoid, load_planar_dataset, load_extra_datasets

np.random.seed(1)  # 设置随机数种子

# 实现含有一个隐藏层的神经网络，你将会体验到与之前logistic实现的不同：
#   使用含有一个隐藏层的神经网络实现2分类。
#   使用一个非线性的激活函数(比如tanh)。
#   计算交叉熵损失。
#   实现前向传播和反向传播。

# 数据集
X, Y = load_planar_dataset()

shape_X = X.shape
shape_Y = Y.shape
m = shape_X[1]  # 训练集样本个数

print ('The shape of X is: ' + str(shape_X))
print ('The shape of Y is: ' + str(shape_Y))
print ('I have m = %d training examples!' % (m))

def lg_sklearn():
    # 简单的逻辑回归
    # 在建立全连接之前，我们首先来看一下逻辑回归对于该问题的表现，可以使用sklearn的内建函数来实现
    clf = sklearn.linear_model.LogisticRegressionCV()
    clf.fit(X.T, Y.T.ravel())
    # numpy中的ravel()、flatten()、squeeze()都有将多维数组转换为一维数组的功能，区别：
    # ravel()：如果没有必要，不会产生源数据的副本
    # flatten()：返回源数据的副本
    # squeeze()：只能对维数为1的维度降维

    plot_decision_boundary(lambda x: clf.predict(x), X, Y)
    plt.title('logistic regression')

    LR_predictions = clf.predict(X.T)
    print('Accuracy of logistic regression: %d ' % float((np.dot(Y, LR_predictions) + np.dot(1-Y, 1-LR_predictions)) / float(Y.size)*100) +
          '% ' + "(percentage of correctly labelled datapoints)")
    plt.show()

# 神经网络模型
# 构建神经网络的基本步骤：
#   1. 定义神经网络的结构(输入单元，隐藏单元等)
#   2. 初始化模型参数
#   3. 循环
#       执行前向传播
#       计算损失
#       执行反向传播
#       更新参数(梯度下降)


# ------------------- 定义神经网络结构 -------------------
# n_x: 输入层单元数目, 等于样本特征维度
# n_h: 隐藏层单元数目
# n_y: 输出层单元数目
def layer_sizes(X, Y):
    n_x = X.shape[0]
    n_h = 4
    n_y = Y.shape[0]
    return n_x, n_h, n_y

# 测试
# X_assess, Y_assess = layer_sizes_test_case()
# (n_x, n_h, n_y) = layer_sizes(X_assess, Y_assess)
# print("The size of the input layer is: n_x = " + str(n_x))
# print("The size of the hidden layer is: n_h = " + str(n_h))
# print("The size of the output layer is: n_y = " + str(n_y))


# ------------------- 初始化模型参数 -------------------
def initialize_parameters(n_x, n_h, n_y):
    np.random.seed(2)

    W1 = np.random.randn(n_h, n_x) * 0.01  # W shape (本层神经元个数, 上一层神经元个数)
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))

    parameters = {'W1': W1,
                  'b1': b1,
                  'W2': W2,
                  'b2': b2}
    return parameters


# n_x, n_h, n_y = initialize_parameters_test_case()
# parameters = initialize_parameters(n_x, n_h, n_y)
# print("W1 = " + str(parameters["W1"]))
# print("b1 = " + str(parameters["b1"]))
# print("W2 = " + str(parameters["W2"]))
# print("b2 = " + str(parameters["b2"]))


# ------------------- 循环: 前向传播 -------------------
# 前向传播的计算，计算过程中需要进行缓存，缓存会作为方向传播的输入
def forward_propagation(X, parameters):
    W1 = parameters['W1']
    b1 = parameters['b1']
    W2 = parameters['W2']
    b2 = parameters['b2']

    # 前向传播
    Z1 = np.dot(W1, X) + b1
    A1 = np.tanh(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = sigmoid(Z2)
    assert (A2.shape == (1, X.shape[1]))
    cache = {'Z1': Z1,
             'A1': A1,
             'Z2': Z2,
             'A2': A2}
    return A2, cache

# 测试
# X_assess, parameters = forward_propagation_test_case()
# A2, cache = forward_propagation(X_assess, parameters)
# # Note: we use the mean here just to make sure that your output matches ours.
# print(np.mean(cache['Z1']) ,np.mean(cache['A1']),np.mean(cache['Z2']),np.mean(cache['A2']))


# ------------------- 循环: 交叉熵 -------------------
def compute_cost(A2, Y, parameters):
    """
    :param A2: 预测值
    :param Y: 真实值
    :param parameters: 参数map
    :return:
    """
    m = Y.shape[1]
    logprobs = np.multiply(np.log(A2), Y) + np.multiply(np.log(1-A2), (1-Y))
    cost = -1/m * np.sum(logprobs)
    cost = np.squeeze(cost)  # makes sure cost is the dimension we expect.
                             # E.g., turns [[17]] into 17
    assert (isinstance(cost, float))
    return cost

# 测试
A2, Y_assess, parameters = compute_cost_test_case()
print("cost = " + str(compute_cost(A2, Y_assess, parameters)))


# ------------------- 循环: 反向传播 -------------------
def backward_propagation(parameters, cache, X, Y):
    """
    :param parameters: 参数map
    :param cache: 前向传播过程中保存到的z，a
    :param X: 输入数据
    :param Y: 输出数据
    :return:
    """
    m = X.shape[1]
    W1 = parameters['W1']
    W2 = parameters['W2']
    A1 = cache['A1']
    A2 = cache['A2']

    dZ2 = A2 - Y  # (n_y,1)
    dW2 = 1 / m * np.dot(dZ2, A1.T)  # (n_y, 1) .* (1, n_h)
    db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2))
    dW1 = 1 / m * np.dot(dZ1, X.T)
    db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)
    grads = {'dW1': dW1,
             'db1': db1,
             'dW2': dW2,
             'db2': db2}

    return grads


# ------------------- 循环: 参数更新 -------------------
def update_parameters(parameters, grads, learning_rate=1.2):
    # Retrieve each parameter from the dictionary "parameters"
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]

    # Retrieve each gradient from the dictionary "grads"
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]

    # Update rule for each parameter
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    return parameters


# ------------------ 组合 ------------------------
# GRADED FUNCTION: nn_model
def nn_model(X, Y, n_h, num_iterations = 10000, print_cost=False):
    np.random.seed(3)
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]

    # Initialize parameters, then retrieve W1, b1, W2, b2. Inputs: "n_x, n_h, n_y". Outputs = "W1, b1, W2, b2, parameters".
    parameters = initialize_parameters(n_x, n_h, n_y)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    # Loop (gradient descent)
    for i in range(0, num_iterations):
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        # Cost function. Inputs: "A2, Y, parameters". Outputs: "cost".
        cost = compute_cost(A2, Y, parameters)
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
        # Gradient descent parameter update. Inputs: "parameters, grads". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate = 1.2)
        # Print the cost every 1000 iterations
        if print_cost and i % 1000 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters