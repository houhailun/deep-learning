#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/10/15 19:15
# Author: Hou hailun

import tensorflow as tf
import numpy as np

rng = np.random
print(tf.__version__)

# 参数
learning_rate = 0.01
train_steps = 1000
display_step = 50

# 训练集
X = np.array([3.3,4.4,5.5,6.71,6.93,4.168,9.779,6.182,7.59,2.167,
              7.042,10.791,5.313,7.997,5.654,9.27,3.1])
Y = np.array([1.7,2.76,2.09,3.19,1.694,1.573,3.366,2.596,2.53,1.221,
              2.827,3.465,1.65,2.904,2.42,2.94,1.3])

W = tf.Variable(rng.randn(), name='weight')
b = tf.Variable(rng.randn(), name='bias')


def linear_regression(x):
    return W * x + b


def mean_square(y_pred, y_true):
    # mse
    return tf.reduce_mean(tf.square(y_pred - y_true))


# 随机梯度下降优化器
optimizer = tf.optimizers.SGD(learning_rate=learning_rate)


# API详解
# tf.GradientTape(persistent=False,watch_accessed_variables=True): 梯度带函数
#   作用：创建一个新的GradientTape
#   persistent：布尔值，用来指定新创建的gradient tape是否是可持续性的。默认是False，意味着只能够调用一次gradient（）函数。
#   watch_accessed_variables: 布尔值，表明这个gradien tap是不是会自动追踪任何能被训练（trainable）的变量。默认是True。要是为False的话，意味着你需要手动去指定你想追踪的那些变量。

# gradient(target,sources,output_gradients=None,unconnected_gradients=tf.UnconnectedGradients.NONE)
#   作用: 根据tape上面的上下文来计算某个或者某些tensor的梯度
#   persistent：布尔值，用来指定新创建的gradient tape是否是可持续性的。默认是False，意味着只能够调用一次gradient（）函数。
#   watch_accessed_variables: 布尔值，表明这个gradien tap是不是会自动追踪任何能被训练（trainable）的变量。默认是True。要是为False的话，意味着你需要手动去指定你想追踪的那些变量。

# apply_gradients(grads_and_vars,name=None)
#   作用:把计算出来的梯度更新到变量上面去


def run_optimization():
    with tf.GradientTape() as g:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)

    gradients = g.gradient(loss, [W, b])               # 计算loss对W，对b的梯度
    optimizer.apply_gradients(zip(gradients, [W, b]))  # 用计算出的梯度更新W，b


for step in range(1, train_steps + 1):
    run_optimization()

    if step % display_step == 0:
        pred = linear_regression(X)
        loss = mean_square(pred, Y)
        print(print("step: %i, loss: %f, W: %f, b: %f" % (step, loss, W.numpy(), b.numpy())))

import matplotlib.pyplot as plt

plt.plot(X, Y, 'ro', label='Original data')
plt.plot(X, np.array(W * X + b), label='Fitted line')
plt.legend()
plt.show()