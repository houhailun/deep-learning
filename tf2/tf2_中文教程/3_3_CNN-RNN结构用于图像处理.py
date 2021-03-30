#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2021/3/24 11:22
# Author: Hou hailun

import numpy as np
from sklearn.metrics import accuracy_score
from tensorflow import keras
from tensorflow.keras import layers


# 1、导入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

# 类别向量转换为二进制（只有0和1）的矩阵类型表示。其表现为将原有的类别向量转换为独热编码的形式
y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)

# 2、简单的cnn-rnn结构
model = keras.Sequential()
x_shape = x_train.shape
model.add(layers.Conv2D(input_shape=(x_shape[1], x_shape[2], x_shape[3]),
                        filters=32, kernel_size=(3, 3), strides=(1, 1),
                        padding='same', activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))
print(model.output_shape)

model.add(layers.Reshape(target_shape=(16*16, 32)))
model.add(layers.LSTM(50, return_sequences=False))
model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(),
             loss=keras.losses.CategoricalCrossentropy(),
             metrics=['accuracy'])
model.summary()

history = model.fit(x_train, y_train, batch_size=32,epochs=5, validation_split=0.1)

