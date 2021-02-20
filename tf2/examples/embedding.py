#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/10/12 17:52
# Author: Hou hailun

# embedding
#   Q1: 什么是embedding?
#   A1: 在推荐系统中，对于离散特征通常转换为one-hot，造成特征维度很大，数据稀疏，针对这种问题，需要做embedding处理；
#   Q2: embedding的过程是什么样子?
#   A2: 实际上是一层全连接神经网络，由于one-hot后只有1个位置是1，其他是0，因此得到的embedding就是1上与下一层的的权重

import tensorflow as tf

# embedding
embedding = tf.constant([[0.21,0.41,0.51,0.11],
                        [0.22,0.42,0.52,0.12],
                        [0.23,0.43,0.53,0.13],
                        [0.24,0.44,0.54,0.14]], dtype=tf.float32)

feature_batch = tf.constant([2, 3, 1, 0])

# embedding_lookup 使用矩阵相乘实现，可以看作特殊的全连接层
get_embedding1 = tf.nn.embedding_lookup(embedding, feature_batch)