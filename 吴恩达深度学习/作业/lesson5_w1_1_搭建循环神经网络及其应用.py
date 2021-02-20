#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2021/2/7 14:20
# Author: Hou hailun

"""
主要内容：一步步搭建循环神经网络
1 - 循环神经网络的前向传播
    1.1 - RNN单元
    1.2 - RNN的前向传播

2 - 长短时记忆（Long Short-Term Memory (LSTM)）网络
    “门”的介绍
        2.0.1 - 遗忘门
        2.0.2 - 更新门
        2.0.3 - 更新单元
        2.0.4 - 输出门
    2.1 - LSTM单元
    2.2 - LSTM的前向传播

3 - 循环神经网络的反向传播（选学）
    3.1 - 基本的RNN网络的反向传播
    3.2 - LSTM反向传播
        3.2.1 - 单步反向传播
        3.2.2 门的导数
        3.2.3参数的导数
    3.3 - LSTM网络的反向传播

字符级语言模型 - 恐龙岛
    1 - 问题描述
        1.1 - 数据集与预处理
        1.2 - 模型回顾
    2 - 构建模型中的模块
        2.1 梯度修剪
        2.2 - 采样
    3 - 构建语言模型
        3.1 - 梯度下降
        3.2 - 训练模型
4 - 写出莎士比亚风格的文字（选学）

用LSTM网络即兴独奏爵士乐
    1 - 问题描述
        1.1 - 数据集
        1.2 - 模型预览
    2 - 构建模型
    3 - 生成音乐
        3.1 - 预测与采样
        3.3 - 生成音乐
"""

import numpy as np
from utils import rnn_utils


# 1、循环神经网络的前向传播
# 1.1 RNN单元
def rnn_cell_forward(xt, a_pred, parameters):
    """
    根据图2实现RNN单元的单步前向传播

    参数：
        xt -- 时间步“t”输入的数据，维度为（n_x, m）
        a_prev -- 时间步“t - 1”的隐藏隐藏状态，维度为（n_a, m）
        parameters -- 字典，包含了以下内容:
                        Wax -- 矩阵，输入乘以权重，维度为（n_a, n_x）
                        Waa -- 矩阵，隐藏状态乘以权重，维度为（n_a, n_a）
                        Wya -- 矩阵，隐藏状态与输出相关的权重矩阵，维度为（n_y, n_a）
                        ba  -- 偏置，维度为（n_a, 1）
                        by  -- 偏置，隐藏状态与输出相关的偏置，维度为（n_y, 1）

    返回：
        a_next -- 下一个隐藏状态，维度为（n_a， m）
        yt_pred -- 在时间步“t”的预测，维度为（n_y， m）
        cache -- 反向传播需要的元组，包含了(a_next, a_prev, xt, parameters)
    """
    Wax = parameters['Wax']
    Waa = parameters['Waa']
    Wya = parameters['Wya']
    ba = parameters['ba']
    by = parameters['by']

    # 计算下一时间步的激活值
    a_next = np.tanh(np.dot(Waa, a_pred) + np.dot(Wax, xt) + ba)

    # 计算当前单元的输出
    yt_pred = rnn_utils.softmax(np.dot(Wya, a_next) + by)

    # 保存反向传播需要的值
    cache = (a_next, a_pred, xt, parameters)

    return a_next, yt_pred, cache

# np.random.seed(1)
# xt = np.random.randn(3,10)
# a_prev = np.random.randn(5,10)
# Waa = np.random.randn(5,5)
# Wax = np.random.randn(5,3)
# Wya = np.random.randn(2,5)
# ba = np.random.randn(5,1)
# by = np.random.randn(2,1)
# parameters = {"Waa": Waa, "Wax": Wax, "Wya": Wya, "ba": ba, "by": by}
#
# a_next, yt_pred, cache = rnn_cell_forward(xt, a_prev, parameters)
# print("a_next[4] = ", a_next[4])
# print("a_next.shape = ", a_next.shape)
# print("yt_pred[1] =", yt_pred[1])
# print("yt_pred.shape = ", yt_pred.shape)


# 1.2 RNN的前向传播
def rnn_forward(x, a0, parameters):
    """
    根据图3来实现循环神经网络的前向传播

    参数：
        x -- 输入的全部数据，维度为(n_x, m, T_x)
        a0 -- 初始化隐藏状态，维度为 (n_a, m)
        parameters -- 字典，包含了以下内容:
                        Wax -- 矩阵，输入乘以权重，维度为（n_a, n_x）
                        Waa -- 矩阵，隐藏状态乘以权重，维度为（n_a, n_a）
                        Wya -- 矩阵，隐藏状态与输出相关的权重矩阵，维度为（n_y, n_a）
                        ba  -- 偏置，维度为（n_a, 1）
                        by  -- 偏置，隐藏状态与输出相关的偏置，维度为（n_y, 1）

    返回：
        a -- 所有时间步的隐藏状态，维度为(n_a, m, T_x)
        y_pred -- 所有时间步的预测，维度为(n_y, m, T_x)
        caches -- 为反向传播的保存的元组，维度为（【列表类型】cache, x)）
    """
    # 初始化“caches”，它将以列表类型包含所有的cache
    caches = []

    # 获取 x 与 Wya 的维度信息
    n_x, m, T_x = x.shape
    n_y, n_a = parameters['Wya'].shape

    # 使用0来初始化“a” 与“y”
    a = np.zeros([n_a, m, T_x])
    y_pred = np.zeros([n_y, m, T_x])

    # 初始化next
    a_next = a0

    # 遍历所有时间步
    for t in range(T_x):
        # 1、使用rnn_cell_forware()来更新next隐藏状态与cache
        a_next, yt_pred, cache = rnn_cell_forward(x[:, :, t], a_next, parameters)

        # 2、使用 a 来保存“next”隐藏状态（第 t ）个位置。
        a[:, :, t] = a_next

        # 3、使用 y 来保存预测值。
        y_pred[:, :, t] = yt_pred

        # 4、把cache保存到“caches”列表中。
        caches.append(cache)

    return a, y_pred, caches


# 2、长短时记忆LSTM模型
# 为了解决rnn的梯度消失问题，增加了记忆细胞cell和3个门控
# 2.1 LSTM单元
def lstm_cell_forward(xt, a_prev, c_prev, parameters):
    """
    根据图4实现一个LSTM单元的前向传播。

    参数：
        xt -- 在时间步“t”输入的数据，维度为(n_x, m)
        a_prev -- 上一个时间步“t-1”的隐藏状态，维度为(n_a, m)
        c_prev -- 上一个时间步“t-1”的记忆状态，维度为(n_a, m)
        parameters -- 字典类型的变量，包含了：
                        Wf -- 遗忘门的权值，维度为(n_a, n_a + n_x)
                        bf -- 遗忘门的偏置，维度为(n_a, 1)
                        Wi -- 更新门的权值，维度为(n_a, n_a + n_x)
                        bi -- 更新门的偏置，维度为(n_a, 1)
                        Wc -- 第一个“tanh”的权值，维度为(n_a, n_a + n_x)
                        bc -- 第一个“tanh”的偏置，维度为(n_a, n_a + n_x)
                        Wo -- 输出门的权值，维度为(n_a, n_a + n_x)
                        bo -- 输出门的偏置，维度为(n_a, 1)
                        Wy -- 隐藏状态与输出相关的权值，维度为(n_y, n_a)
                        by -- 隐藏状态与输出相关的偏置，维度为(n_y, 1)
    返回：
        a_next -- 下一个隐藏状态，维度为(n_a, m)
        c_next -- 下一个记忆状态，维度为(n_a, m)
        yt_pred -- 在时间步“t”的预测，维度为(n_y, m)
        cache -- 包含了反向传播所需要的参数，包含了(a_next, c_next, a_prev, c_prev, xt, parameters)

    注意：
        ft/it/ot表示遗忘/更新/输出门，cct表示候选值(c tilda)，c表示记忆值。
    """
    # 从“parameters”中获取相关值
    Wf = parameters["Wf"]
    bf = parameters["bf"]
    Wi = parameters["Wi"]
    bi = parameters["bi"]
    Wc = parameters["Wc"]
    bc = parameters["bc"]
    Wo = parameters["Wo"]
    bo = parameters["bo"]
    Wy = parameters["Wy"]
    by = parameters["by"]

    # 获取 xt 与 Wy 的维度信息
    n_x, m = xt.shape
    n_y, n_a = Wy.shape

    # 1、链接 a_prev 与 xt
    concat = np.zeros([n_a + n_x, m])  # 行数增加，列不变，即样本不变，维度增加
    concat[: n_a, :] = a_prev          # 前na行数据为i激活值a，设置为前一个时间步的激活值
    concat[n_a:, :] = xt               # 下面的数据为x，设置为当前时间步的输入x

    # 2、根据公式计算ft,it,cct,c_next,ot,a_next
    # 遗忘门
    ft = rnn_utils.sigmoid(np.dot(Wf, concat) + bf)

    # 更新们
    it = rnn_utils.sigmoid(np.dot(Wi, concat) + bi)

    # 候选集
    cct = np.tanh(np.dot(Wc, concat) + bc)

    # 更新记忆
    c_next = ft * c_prev + it * cct

    # 输出门
    ot = rnn_utils.sigmoid(np.dot(Wo, concat) + bo)

    # 计算下一个激活值
    a_next = ot * np.tanh(c_next)

    # 预测输出
    yt_pred = rnn_utils.softmax(np.dot(Wy, a_next) + by)

    # 保存包含了反向传播所需要的参数
    cache = (a_next, c_next, a_prev, c_prev, ft, it, cct, ot, xt, parameters)

    return a_next, c_next, yt_pred, cache


# 2.2 LSTM的前向传播
def lstm_forward(x, a0, parameters):
    """
    根据图5来实现LSTM单元组成的的循环神经网络

    参数：
        x -- 所有时间步的输入数据，维度为(n_x, m, T_x)
        a0 -- 初始化隐藏状态，维度为(n_a, m)
        parameters -- python字典，包含了以下参数：
                        Wf -- 遗忘门的权值，维度为(n_a, n_a + n_x)
                        bf -- 遗忘门的偏置，维度为(n_a, 1)
                        Wi -- 更新门的权值，维度为(n_a, n_a + n_x)
                        bi -- 更新门的偏置，维度为(n_a, 1)
                        Wc -- 第一个“tanh”的权值，维度为(n_a, n_a + n_x)
                        bc -- 第一个“tanh”的偏置，维度为(n_a, n_a + n_x)
                        Wo -- 输出门的权值，维度为(n_a, n_a + n_x)
                        bo -- 输出门的偏置，维度为(n_a, 1)
                        Wy -- 隐藏状态与输出相关的权值，维度为(n_y, n_a)
                        by -- 隐藏状态与输出相关的偏置，维度为(n_y, 1)

    返回：
        a -- 所有时间步的隐藏状态，维度为(n_a, m, T_x)
        y -- 所有时间步的预测值，维度为(n_y, m, T_x)
        caches -- 为反向传播的保存的元组，维度为（【列表类型】cache, x)）
    """
    caches  = []

    # 获取 xt 与 Wy 的维度信息
    n_x, m, T_x = x.shape
    n_y, n_a = parameters["Wy"].shape

    # 使用0来初始化a, c, y
    a = np.zeros([n_a, m, T_x])
    c = np.zeros([n_a, m, T_x])
    y = np.zeros([n_y, m, T_x])

    # 初始化“a_next”、“c_next”
    a_next = a0
    c_next = np.zeros([n_a, m])

    # 遍历所有时间步t
    for t in range(T_x):
        # 更新下一个隐藏状态，下一个记忆状态，计算预测值，获取cache
        a_next, c_next, yt_pred, cache = lstm_cell_forward(x[:, :, t], a_next, c_next, parameters)

        # 保存新的下一个隐藏状态到变量a中
        a[:, :, t] = a_next

        # 保存预测值到变量y中
        y[:, :, t] = yt_pred

        # 保存下一个单元状态到变量c中
        c[:, :, t] = c_next

        # 把cache添加到caches中
        caches.append(cache)

    # 保存反向传播需要的参数
    caches = (caches, x)

    return a, y, c, caches

# np.random.seed(1)
# x = np.random.randn(3,10,7)
# a0 = np.random.randn(5,10)
# Wf = np.random.randn(5, 5+3)
# bf = np.random.randn(5,1)
# Wi = np.random.randn(5, 5+3)
# bi = np.random.randn(5,1)
# Wo = np.random.randn(5, 5+3)
# bo = np.random.randn(5,1)
# Wc = np.random.randn(5, 5+3)
# bc = np.random.randn(5,1)
# Wy = np.random.randn(2,5)
# by = np.random.randn(2,1)
#
# parameters = {"Wf": Wf, "Wi": Wi, "Wo": Wo, "Wc": Wc, "Wy": Wy, "bf": bf, "bi": bi, "bo": bo, "bc": bc, "by": by}
#
# a, y, c, caches = lstm_forward(x, a0, parameters)
# print("a[4][3][6] = ", a[4][3][6])
# print("a.shape = ", a.shape)
# print("y[1][4][3] =", y[1][4][3])
# print("y.shape = ", y.shape)
# print("caches[1][1[1]] =", caches[1][1][1])
# print("c[1][2][1]", c[1][2][1])
# print("len(caches) = ", len(caches))


# 3、循环神经网络的反向传播
def rnn_cell_backward(da_next, cache):
    """
    实现基本的RNN单元的单步反向传播

    参数：
        da_next -- 关于下一个隐藏状态的损失的梯度。
        cache -- 字典类型，rnn_step_forward()的输出

    返回：
        gradients -- 字典，包含了以下参数：
                        dx -- 输入数据的梯度，维度为(n_x, m)
                        da_prev -- 上一隐藏层的隐藏状态，维度为(n_a, m)
                        dWax -- 输入到隐藏状态的权重的梯度，维度为(n_a, n_x)
                        dWaa -- 隐藏状态到隐藏状态的权重的梯度，维度为(n_a, n_a)
                        dba -- 偏置向量的梯度，维度为(n_a, 1)
    """
    # 获取cache 的值
    a_next, a_prev, xt, parameters = cache

    # 从 parameters 中获取参数
    Wax = parameters["Wax"]
    Waa = parameters["Waa"]
    Wya = parameters["Wya"]
    ba = parameters["ba"]
    by = parameters["by"]

    # 计算tanh相对于a_next的梯度
    dtanh = (1 - np.squeeze(a_next)) * da_next

    # 计算关于Wax损失的梯度
    dxt = np.dot(Wax.T, dtanh)
    dWax = np.dot(dtanh, xt.T)

    # 计算关于Waa损失的梯度
    da_prev = np.dot(Waa.T, dtanh)
    dWaa = np.dot(dtanh, a_prev.T)

    # 计算关于b损失的梯度
    dba = np.sum(dtanh, keepdims=True, axis=-1)

    # 保存这些梯度到字典内
    gradients = {"dxt": dxt, "da_prev": da_prev, "dWax": dWax, "dWaa": dWaa, "dba": dba}

    return gradients