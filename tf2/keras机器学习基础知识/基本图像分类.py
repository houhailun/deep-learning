#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/10/15 16:14
# Author: Hou hailun

import tensorflow as tf
from tensorflow import keras

import numpy as np
import matplotlib.pyplot as plt

print(tf.__version__)

fashion_mnist = keras.datasets.fashion_mnist
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()

