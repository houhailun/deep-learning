#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/11/25 15:48
# Author: Hou hailun

# BP算法 python实现

import numpy as np


def nonlin(x, deriv=False):
    # deriv为真，则求梯度；否则求sigmod
    if deriv:
        return x * (1 - x)
    return 1 / (1 + np.exp(-x))

# 定义输入输出
X = np.array([[0.35], [0.9]])
y = np.array([[0.5]])

np.random.seed(1)

W0 = np.array([[0.1, 0.8], [0.4, 0.6]])
W1 = np.array([[0.3, 0.9]])

print('original ', W0, '\n', W1)
for j in range(100):  # 迭代100次
    # 定义前向传播
    l0 = X
    l1 = nonlin(np.dot(W0, l0))
    l2 = nonlin(np.dot(W1, l1))
    l2_error = y - l2
    error = 1 / 2.0 * (y - l2) ** 2
    print('Error:', error)

    # 定义反向传播
    l2_delta = l2_error * nonlin(l2, deriv=True)  # this will backpack
    l1_error = l2_delta * W1  # 反向传播
    l1_delta = l1_error * nonlin(l1, deriv=True)

    W1 += l2_delta * l1.T  # 修改权值
    W0 += l0.T.dot(l1_delta)
    print(W0, '\n', W1)