#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2021/3/12 15:03
# Author: Hou hailun

import tensorflow as tf
print(tf.__version__)

# 在tensorflow中，tensor是一个类，也是存储和变换数据的主要工具。如果你之前用过NumPy，你会发现tensor和NumPy的多维数组非常类似。然而，tensor提供GPU计算和自动求梯度等更多功能，这些使tensor更加适合深度学习

# 2.2.1 create ndarray
x = tf.constant(value=range(12))
print(x.shape)
print(x)
print(len(x))

# reshape转换tensor形状
X = tf.reshape(tensor=x, shape=(3, 4))
print(X)

print(tf.zeros(shape=(2,3,4)))
print(tf.ones(shape=(3,4)))
Y = tf.constant([[2,1,4,3],[1,2,3,4],[4,3,2,1]])
print(tf.random.normal(shape=[3,4], mean=0, stddev=1))


# 2.2.2 arithmetic ops
print(X+Y)




