#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/10/14 10:34
# Author: Hou hailun

# 使用tf构建KMeans模型
from __future__ import print_function

import numpy as np
import tensorflow as tf
from tensorflow.contrib.factorization import KMeans

# import mnist data
from tensorflow.examples.tutorials.mnist import input_data
minist = input_data.read_data_sets("/tmp/data", one_hot=True)
full_data_x = minist.images

