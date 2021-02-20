#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/10/15 19:39
# Author: Hou hailun

# 使用tensorflow构建LR回归模型

import tensorflow as tf
import numpy as np

# minist dataset param
num_classes = 10     # 类别个数
num_features = 784   # 特征个数

# 训练参数
learning_rate = 0.01
training_step = 1000
batch_size = 256
display_step = 50

# 训练集
from tensorflow.keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# 转换为float32
x_train, x_test = np.array(x_train, np.float32), np.array(x_test, np.float32)
# 28*28 转换为1维 1*184
x_train, x_test = x_train.reshape([-1, num_features]), x_test.reshape([-1, num_features])
# 标准化
x_train, x_test = x_train / 255, x_test / 255

# Use tf.data API to shuffle and batch data.
train_data = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_data = train_data.repeat().shuffle(5000).batch(batch_size).prefetch(1)

# weight: [784, 10]
W = tf.Variable(tf.ones([num_features, num_classes]), name='weight')
b = tf.Variable(tf.zeros([num_classes]), name='bias')


def logistic_regression(x):
    return tf.nn.softmax(tf.matmul(x, W) + b)


def cross_entropy(y_pred, y_true):
    y_true = tf.one_hot(y_true, depth=num_classes)
    y_pred = tf.clip_by_value(y_pred, 1e-9, 1.0)  # 数据压缩到1e-9~1.0之间
    return tf.reduce_mean(-tf.reduce_sum(y_true * tf.math.log(y_pred), 1))
