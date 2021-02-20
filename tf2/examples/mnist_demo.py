#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/2/24 17:05
# Author: Hou hailun

import os
import sys
import numpy as np
import tensorflow as tf

import input_data

print('python version:', sys.version)
print('tf version:', tf.__version__)

cur_path = os.getcwd()
# 数据集
mnist = input_data.read_data_sets("datasets/", one_hot=True)


def softmax():
    # softmax 回归
    x = tf.placeholder('float', [None, 784])  # None表示张量的第一个维度可以是任何长度的
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))
    y = tf.nn.softmax(tf.matmul(x, W) + b)

    # 损失函数
    y_ = tf.placeholder('float', [None, 10])        # 真实值
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))  # 交叉熵

    # 优化算法
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)  # 梯度下降算法最小化交叉熵

    # 初始化变量
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    # 训练模型
    for i in range(1000):
        batch_xs, batch_ys = mnist.train.next_batch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # 评估模型
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # axis=1将每一行最大元素所在的索引记录下来，最后返回每一行最大元素所在的索引数组。
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))
    print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


class CNN:
    # CNN 卷积神经网络
    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
