#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/2/24 15:54
# Author: Hou hailun

import tensorflow as tf
import numpy as np

print(tf.__version__)


def line_demo():
    # 使用 NumPy 生成假数据(phony data), 总共 100 个点.
    x_data = np.float32(np.random.rand(2, 100))      # 随机输入
    y_data = np.dot([0.100, 0.200], x_data) + 0.300  # 1*100

    # 构造一个线性模型
    b = tf.Variable(tf.zeros([1]))
    W = tf.Variable(tf.random_uniform([1, 2], -1.0, 1.0))
    y = tf.matmul(W, x_data) + b

    # 最小化方差
    loss = tf.reduce_mean(tf.square(y - y_data))        # 损失函数为均方差
    optimizer = tf.train.GradientDescentOptimizer(0.5)  # 优化算法
    train = optimizer.minimize(loss)                    # 优化：最小化损失函数

    # 初始化变量
    init = tf.global_variables_initializer()

    # 启动图
    sess = tf.Session()
    sess.run(init)

    # 拟合平面
    for step in range(0, 201):
        sess.run(train)
        if step % 20 == 0:
            print(step, sess.run(W), sess.run(b))

# line_demo()

def test():
    # 构建图
    matrix1 = tf.constant([[3.0, 3.0]])
    matrix2 = tf.constant([[2.0], [2.0]])
    product = tf.matmul(matrix1, matrix2)

    # Fetch: 取回操作的输出内容
    input1 = tf.constant(3.0)
    input2 = tf.constant(2.0)
    input3 = tf.constant(5.0)
    intermed = tf.add(input2, input3)
    mul = tf.multiply(input1, intermed)

    with tf.Session() as sess:
      result = sess.run([mul, intermed])
      print(result)


    # Feed
    # 该机制可以临时替代图中的任意操作中的 tensor 可以对图中任何操作提交补丁, 直接插入一个 tensor.
    input1 = tf.placeholder(tf.float32)  # 占位符
    input2 = tf.placeholder(tf.float32)
    output = tf.multiply(input1, input2)

    with tf.Session() as sess:
      print(sess.run([output], feed_dict={input1: [7.0],
                                          input2: [2.0]}))

def softmax():
    x = np.array([[-3.1, 1.8, 9.7, -2.5]])
    pred = tf.nn.softmax(x)
    print(pred)  # Tensor("Softmax:0", shape=(1, 4), dtype=float64)

    sess = tf.Session()
    print(sess.run(pred))
    sess.close()
print(softmax())
