#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2021/3/11 17:49
# Author: Hou hailun

import tensorflow as tf
from tensorflow import keras

model = keras.models.Sequential()
model.add(keras.layers.Flatten(input_shape=[28, 28]))
for _ in range(10):
    model.add(keras.layers.Dense(units=30, activation="relu"))

model.add(keras.layers.Dense(units=10, activation="softmax"))

