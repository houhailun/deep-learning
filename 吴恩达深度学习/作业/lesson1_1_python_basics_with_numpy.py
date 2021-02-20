#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/10/14 9:31
# Author: Hou hailun

# 使用numpy构建基本的函数

import math
import numpy as np


# ------------------------- sigmod() -------------------------
def basic_sigmoid(x):
    s = 1 / (1 + math.exp(-x))
    return s


# 实际上深度学习往往是矩阵和向量形式，math库的函数的参数往往是一个数值，深度学习中一版使用numpy库
def sigmoid(x):
    s = 1 / (1 + np.exp(-x))
    return s


# ------------------------- sigmod的梯度: s(1-s) -------------------------
def sigmoid_derivative(x):
    s = 1 / (1 + np.exp(-x))
    return s * (1 - s)
# print(sigmoid_derivative(np.array([1,2,3])))


# ------------------------- reshape arrays -------------------------
# v = v.reshape((v.shape[0] * v.shape[1], v.shape[2]))  # 转换矩阵为向量
def image2vector(image):
    # 将三维图像矩阵调整为向量
    v = image.reshape((image.shape[0] * image.shape[1] * image.shape[2], 1))
    return v


# ------------------------- normalizing rows -------------------------
def normalizeRows(x):
    # l2正则
    x_norm = np.linalg.norm(x, ord=2, axis=1, keepdims=True)  # 按行求l2
    x = x / x_norm
    return x


# ------------------------- boradcasting and the softmax -------------------------
def softmax(x):
    x_exp = np.exp(x)
    x_sum = np.sum(x_exp, axis=1, keepdims=True)  # 按行求和, keepdims保持矩阵维度
    s = x / x_sum
    return s

x = np.array([[9, 2, 5, 0, 0], [7, 5, 0, 0, 0]])
# print(softmax(x))


# ------------------------- vectorization -------------------------
# 一维向量点积运算
# 非向量化
import time
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

tic = time.process_time()
dot = 0
for i in range(len(x1)):
    dot += x1[i] * x2[i]
toc = time.process_time()
print("dot = " + str(dot) + "\n ----- Computation time = " + str(1000 * (toc - tic)) + "ms")

# 向量化
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

tic = time.process_time()
dot = np.dot(x1, x2)
toc = time.process_time()
print ("dot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")


# ------------------------- outer运算 -------------------------
# outer(): 求外积
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

tic = time.process_time()
outer = np.outer(x1, x2)
toc = time.process_time()
print ("outer = " + str(outer) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")


# ------------------------- 对应位置相乘 -------------------------
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]

tic = time.process_time()
mul = np.multiply(x1, x2)
toc = time.process_time()
print ("elementwise multiplication = " + str(mul) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")


# ------------------------- 矩阵相乘 -------------------------
x1 = [9, 2, 5, 0, 0, 7, 5, 0, 0, 0, 9, 2, 5, 0, 0]
x2 = [9, 2, 2, 9, 0, 9, 2, 5, 0, 0, 9, 2, 5, 0, 0]
W = np.random.rand(3, len(x1))  # Random 3*len(x1) numpy array

tic = time.process_time()
dot = np.dot(W, x1)
toc = time.process_time()
print ("gdot = " + str(dot) + "\n ----- Computation time = " + str(1000*(toc - tic)) + "ms")


# ------------------------- L1/l2 -------------------------
def l1(yhat, y):
    loss = np.sum(np.abs(yhat - y))
    return loss


def l2(yhat, y):
    loss = np.dot((yhat - y), (yhat - y))
    return loss

yhat = np.array([.9, 0.2, 0.1, .4, .9])
y = np.array([1, 0, 0, 1, 1])
print("L2 = " + str(l2(yhat,y)))