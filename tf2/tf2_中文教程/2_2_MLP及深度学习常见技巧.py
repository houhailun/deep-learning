#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2021/3/23 16:09
# Author: Hou hailun

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
print(tf.__version__)

# 1、导入数据
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape([x_train.shape[0], -1])  # (60000, 28, 28) -> (60000, 784)
x_test = x_test.reshape([x_test.shape[0], -1])
print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)

# # 2、基础模型
# model = keras.Sequential([
#     layers.Dense(units=64, activation='relu', input_shape=(784,)),  # 输入维度是784维
#     layers.Dense(units=64, activation='relu'),
#     layers.Dense(units=64, activation='relu'),
#     layers.Dense(units=10, activation='softmax')
# ])
# model.compile(optimizer=keras.optimizers.Adam(),
#               loss=keras.losses.SparseCategoricalCrossentropy(),
#               metrics=['accuracy'])
# model.summary()
#
# history = model.fit(x_train, y_train, batch_size=256, epochs=100, validation_split=0.3, verbose=0)
# print(history)
#
# # import matplotlib.pyplot as plt
# # plt.plot(history.history['accuracy'])
# # plt.plot(history.history['val_accuracy'])
# # plt.legend(['training', 'validation'], loc='upper left')
# # plt.show()
#
# result = model.evaluate(x_test, y_test)
# print(result)

# 3、权重初始化
# model = keras.Sequential([
#     layers.Dense(units=64, activation='relu', kernel_initializer='he_normal', input_shape=(784,)),
#     layers.Dense(units=64, activation='relu', kernel_initializer='he_normal'),
#     layers.Dense(units=64, activation='relu', kernel_initializer='he_normal'),
#     layers.Dense(units=64, activation='softmax')
# ])
# model.compile(optimizer=keras.optimizers.Adam(),
#              loss=keras.losses.SparseCategoricalCrossentropy(),
#              metrics=['accuracy'])
# model.summary()
#
# history = model.fit(x_train, y_train, batch_size=256, epochs=100, validation_split=0.3, verbose=0)
# result = model.evaluate(x_test, y_test)

# 4、激活函数
# 5、优化器
# 6、批正则化
# model = keras.Sequential([
#     layers.Dense(units=64, activation='relu', input_shape=(784,)),
#     layers.BatchNormalization(),
#     layers.Dense(units=64, activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dense(units=64, activation='relu'),
#     layers.BatchNormalization(),
#     layers.Dense(units=10, activation='softmax')
# ])
# model.compile(optimizer=keras.optimizers.Adam(),
#              loss=keras.losses.SparseCategoricalCrossentropy(),
#              metrics=['accuracy'])
# model.summary()
# history = model.fit(x_train, y_train, batch_size=256, epochs=100, validation_split=0.3, verbose=0)
# result = model.evaluate(x_test, y_test)
# print(result)

# 7、dropout
# model = keras.Sequential([
#     layers.Dense(64, activation='relu', input_shape=(784,)),
#     layers.Dropout(0.2),
#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.2),
#     layers.Dense(64, activation='relu'),
#     layers.Dropout(0.2),
#     layers.Dense(10, activation='softmax')
# ])
# model.compile(optimizer=keras.optimizers.SGD(),
#              loss=keras.losses.SparseCategoricalCrossentropy(),
#              metrics=['accuracy'])
# model.summary()

# 8、模型集成
# 使用投票的方法进行模型集成
import numpy as np
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


def mlp_model():
    model = keras.Sequential([
    layers.Dense(64, activation='relu', input_shape=(784,)),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
    ])
    model.compile(optimizer=keras.optimizers.SGD(),
             loss=keras.losses.SparseCategoricalCrossentropy(),
             metrics=['accuracy'])
    return model


model1 = KerasClassifier(build_fn=mlp_model, epochs=100, verbose=0)
model2 = KerasClassifier(build_fn=mlp_model, epochs=100, verbose=0)
model3 = KerasClassifier(build_fn=mlp_model, epochs=100, verbose=0)
model1._estimator_type = "classifier"
model2._estimator_type = "classifier"
model3._estimator_type = "classifier"

ensemble_clf = VotingClassifier(estimators=[
    ('model1', model1), ('model2', model2), ('model3', model3)
], voting='soft')
# Hard Voting Classifier：根据少数服从多数来定最终结果；
# Soft Voting Classifier：将所有模型预测样本为某一类别的概率的平均值作为标准，概率最高的对应的类型为最终的预测结果；

ensemble_clf.fit(x_train, y_train)
y_pred = ensemble_clf.predict(x_test)
print('acc: ', accuracy_score(y_pred, y_test))

# 9、全部使用
from tensorflow.keras import layers

import numpy as np
# KerasClassifier实现了Scikit-Learn 分类器接口
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score


def mlp_model():
    model = keras.Sequential([
        layers.Dense(units=64, activation='relu', kernel_initializer='he_normal', input_shape=(784,)),
        layers.BatchNormalization(),
        layers.Dropout(rate=0.2),
        layers.Dense(units=64, activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(rate=0.2),
        layers.Dense(units=64, activation='relu', kernel_initializer='he_normal'),
        layers.BatchNormalization(),
        layers.Dropout(rate=0.2),
        layers.Dense(units=10, activation='softmax')
    ])

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.SparseCategoricalCrossentropy(),
                  metrics=['accuracy'])
    return model


model1 = KerasClassifier(build_fn=mlp_model, epochs=100, verbose=0)
model2 = KerasClassifier(build_fn=mlp_model, epochs=100, verbose=0)
model3 = KerasClassifier(build_fn=mlp_model, epochs=100, verbose=0)
model4 = KerasClassifier(build_fn=mlp_model, epochs=100, verbose=0)
ensemble_clf = VotingClassifier(estimators=[
    ('model1', model1), ('model2', model2), ('model3', model3),('model4', model4)])

ensemble_clf.fit(x_train, y_train)
y_predict = ensemble_clf.predict(x_test)
print('acc: ', accuracy_score(y_pred, y_test))