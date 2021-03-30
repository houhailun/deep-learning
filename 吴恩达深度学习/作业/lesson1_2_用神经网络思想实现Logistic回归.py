#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/10/15 10:15
# Author: Hou hailun

import numpy as np
import matplotlib.pyplot as plt
import h5py
import scipy
from PIL import Image
from scipy import ndimage
from utils.lr_utils import load_dataset

# 构建一个简单的图像识别算法，该算法可以将图片正确分类为猫和非猫
train_set_x_orig, train_set_y, test_set_x_orig, test_set_y, classes = load_dataset()

# index = 5
# plt.imshow(train_set_x_orig[index])
# print("y = " + str(train_set_y[:, index]) + ", it's a '" + classes[np.squeeze(train_set_y[:, index])].decode("utf-8") + "' picture.")

### START CODE HERE ### (≈ 3 lines of code)
m_train = train_set_x_orig.shape[0]  # (图片个数, 高, 宽, 通道=3)
m_test = test_set_x_orig.shape[0]
num_px = train_set_x_orig.shape[1]
### END CODE HERE ###

print ("Number of training examples: m_train = " + str(m_train))
print ("Number of testing examples: m_test = " + str(m_test))
print ("Height/Width of each image: num_px = " + str(num_px))
print ("Each image is of size: (" + str(num_px) + ", " + str(num_px) + ", 3)")
print ("train_set_x shape: " + str(train_set_x_orig.shape))
print ("train_set_y shape: " + str(train_set_y.shape))
print ("test_set_x shape: " + str(test_set_x_orig.shape))
print ("test_set_y shape: " + str(test_set_y.shape))

# 转换为1维向量: (num_px * num_px * 3, 样本数)
train_set_x_flatten = train_set_x_orig.reshape(train_set_x_orig.shape[0], -1).T  # (12288, 209)
test_set_x_flatten = test_set_x_orig.reshape(test_set_x_orig.shape[0], -1).T

# 标准化数据
train_set_x = train_set_x_flatten / 255
test_set_x = test_set_x_flatten / 255


# 构建算法
# 神经网络主要步骤:
#   1、定义模型结构(如输入特征的数量)
#   2、初始化模型的参数
#   3、循环: 计算当前损失(正向传播)、计算当前梯度(反向传播)、更新参数(梯度下降)

def sigmod(z):
    s = 1 / (1 + np.exp(-z))
    return s
# print(sigmod(np.array([0, 2])))


# 初始化参数
def initialize_with_zeros(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b

dim = 2
w, b = initialize_with_zeros(dim)
print('w = '+ str(w))
print('b = '+ str(b))


def propagate(w, b, X, Y):
    """
    前向和反向传播
    :param w: 权重, a numpy array of size (num_px * num_px * 3, 1)
    :param b: bias, a scalar
    :param X: data of size (num_px * num_px * 3, number of examples)
    :param Y: true "label" vector (containing 0 if non-cat, 1 if cat) of size (1, number of examples)
    :return:
        cost -- negative log-likelihood cost for logistic regression
        dw -- gradient of the loss with respect to w, thus same shape as w
        db -- gradient of the loss with respect to b, thus same shape as b
    """
    m = X.shape[1]

    A = sigmod(np.dot(w.T, X) + b)  # 预测值
    cost = -1 / m * np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A))  # 损失函数

    dw = 1 / m * np.dot(X, (A - Y).T)  # 梯度
    db = 1 / m * np.sum(A - Y)

    assert dw.shape == w.shape
    assert db.dtype == float
    cost = np.square(cost)

    grads = {"dw": dw, "db": db}
    return grads, cost

w, b, X, Y = np.array([[1], [2]]), 2, np.array([[1,2], [3, 4]]), np.array([[1, 0]])
# grads, costs = propagate(w, b, X, Y)
# print(grads)
# print(costs)

def optimize(w, b, X, Y, num_iters, learning_rate, print_cost=False):
    """
    优化参数: 梯度下降法
    Arguments:
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat), of shape (1, number of examples)
    num_iterations -- number of iterations of the optimization loop
    learning_rate -- learning rate of the gradient descent update rule
    print_cost -- True to print the loss every 100 steps

    Returns:
    params -- dictionary containing the weights w and bias b
    grads -- dictionary containing the gradients of the weights and bias with respect to the cost function
    costs -- list of all the costs computed during the optimization, this will be used to plot the learning curve.
    """
    costs = []
    for i in range(num_iters):
        grads, cost = propagate(w, b, X, Y)
        dw = grads['dw']
        db = grads['db']
        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
        if print_cost and i % 100 == 0:
            print("Cost after iter %i: %f" % (i, cost))

    params = {'w': w,
              'b': b}
    grads = {'dw': dw,
             'db': db}

    return params, grads, costs

# params, grads, costs = optimize(w, b, X, Y, num_iters=100, learning_rate=0.09, print_cost=False)
# print(params)
# print(grads)
# print(costs)


def predict(w, b, X):
    """
    预测函数
    w -- weights, a numpy array of size (num_px * num_px * 3, 1)
    b -- bias, a scalar
    X -- data of shape (num_px * num_px * 3, number of examples)
    :return:
    """
    m = X.shape[1]
    Y_pred = np.zeros((1, m))
    w = w.reshape(X.shape[0], 1)   # (num_px * num_px * 3, 1)
    A = sigmod(np.dot(w.T, X) + b)
    for i in range(A.shape[1]):
        if A[0, i] > 0.5:
            Y_pred[0, i] = 1
        else:
            Y_pred[0, i] = 0

    assert (Y_pred.shape == (1, m))
    return Y_pred


# 所有函数合并成一个模型
def model(X_train, Y_train, X_test, Y_test, num_iters=2000, learning_rate=0.5, print_cost=False):
    w, b = np.zeros((X_train.shape[0], 1)), 0
    params, grads, costs = optimize(w, b, X_train, Y_train, num_iters, learning_rate, print_cost)
    w = params['w']
    b = params['b']
    Y_pred_test = predict(w, b, X_test)
    Y_pred_train = predict(w, b, X_train)

    print('train accuracy: {} %'.format(100 - np.mean(np.abs(Y_pred_train - Y_train)) * 100))
    print('test accuracy: {} %'.format(100 - np.mean(np.abs(Y_pred_test - Y_test)) * 100))

    d = {'costs': costs,
         'Y_pred_test': Y_pred_test,
         'Y_pred_train': Y_pred_train,
         'w': w,
         'b': b,
         'learning_rate': learning_rate,
         'num_iters': num_iters}
    return d

d = model(train_set_x, train_set_y, test_set_x, test_set_y, num_iters=2000, learning_rate=0.5, print_cost=True)