#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/10/15 19:08
# Author: Hou hailun

import tensorflow as tf

print(tf.__version__)

a = tf.constant(2)
b = tf.constant(2)
c = tf.constant(5)

add = tf.add(a, b)
sub = tf.subtract(a, b)
mul = tf.multiply(a, b)
div = tf.divide(a, b)

print(add.numpy())  # 获得tensor的值

mean = tf.reduce_mean([a, b, c])
sum = tf.reduce_sum([a, b, c])
print(mean.numpy())
print(sum.numpy())

