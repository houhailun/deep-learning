#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2021/3/12 16:55
# Author: Hou hailun

import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
print(tf.__version__)

#
# ------------------ 回归任务 ------------------
#


def regression_boston_housing():
    # 导入数据
    (x_train, y_train), (x_test, y_test) = keras.datasets.boston_housing.load_data()
    print(x_train.shape, ' ', y_train.shape)
    print(x_test.shape, ' ', y_test.shape)

    # 构建模型
    model = keras.Sequential()
    model.add(layers.Dense(units=32, activation='sigmoid', input_shape=(13,)))
    model.add(layers.Dense(units=32, activation='sigmoid'))
    model.add(layers.Dense(units=32, activation='sigmoid'))
    model.add(layers.Dense(units=1))
    """
    两种构建模型方式等价
    model = keras.Sequential([
        layers.Dense(units=32,  activation='sigmoid', input_shape=(13,)),
        layers.Dense(units=32, activation='sigmoid'),
        layers.Dense(units=32, activation='sigmoid'),
        layers.Dense(units=1)
    ])
    """

    # 模型配置
    model.compile(optimizer=keras.optimizers.SGD(learning_rate=0.1),
                  loss=keras.losses.mean_squared_error,
                  metrics=keras.metrics.mse)
    model.summary()

    # 模型训练
    # validation_split: 从训练集中选择10%样本作为验证集
    model.fit(x_train, y_train, batch_size=50, epochs=50, validation_split=0.1, verbose=0.1)

    # 模型评估
    result = model.evaluate(x_test, y_test)
    print(model.metrics_names)
    print(result)

    # 模型预测
    pred = model.predict(x_test)
    df = pd.DataFrame(y_test, columns=['y_test'])
    df['pred'] = pred
    print(df.head())


# regression_boston_housing()

#
# 分类任务
#
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split

whole_data = load_breast_cancer()
x_data = whole_data.data
y_date = whole_data.target

x_train, x_test, y_train, y_test = train_test_split(x_data, y_date, test_size=0.3, random_state=7)
print(x_train.shape, ' ', y_train.shape)  # (398, 30) | (样本数，特征维度)
print(x_test.shape, ' ', y_test.shape)
print(x_train[:2])
exit()

# 模型构建
model = keras.Sequential([
    layers.Dense(units=32, activation='relu', input_shape=(30,)),
    layers.Dense(units=32, activation='relu'),
    layers.Dense(units=1, activation='sigmoid')
])

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.binary_crossentropy,
              metrics=['accuracy'])
model.summary()

model.fit(x_train, y_train, batch_size=64, epochs=10, verbose=1)

res = model.evaluate(x_test, y_test)
print(model.metrics_names)
print(res)