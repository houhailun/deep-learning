#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2021/3/24 9:39
# Author: Hou hailun

from tensorflow import keras
from tensorflow.keras import layers


# 导入数据
num_words = 30000
maxlen = 200
(x_train, y_train), (x_test, y_test) = keras.datasets.imdb.load_data(path='./imdb.npz', num_words=num_words)
print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)

# pad_sequences
# 将num_samples个文本序列列表 (每个序列为整数列表) 转换成一个 2D Numpy数组，数组形状为 (num_samples, num_timesteps)。如果指定了参数 maxlen 的值，则num_timesteps的值取maxlen的值，否则num_timesteps的值等于最长序列的长度
# 小于maxlen的填充，大于maxlen的截断
# 注意：这个转换为2-D是 [1,2,3,4,5] -> [1 2 3 4 5], 也就是每个单词作为一个时间步输入
x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen, padding='post')
x_test = keras.preprocessing.sequence.pad_sequences(x_test, maxlen, padding='post')
print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)


# 2、LSTM
# def lstm_model():
#     model = keras.Sequential([
#         # input_dim: 词汇表大小
#         # input_length：输入序列长度
#         # output_dim：embedding向量维度
#         layers.Embedding(input_dim=30000, output_dim=32, input_length=maxlen),
#         layers.LSTM(units=32, return_sequences=True),  # return_sequences=True表示返回全部time step 的hidden state的值
#         layers.LSTM(units=1, activation='sigmoid', return_sequences=False)
#     ])
#
#     model.compile(optimizer=keras.optimizers.Adam(),
#                   loss=keras.losses.BinaryCrossentropy(),
#                   metrics=['accuracy'])
#     return model
#
#
# model = lstm_model()
# model.summary()
#
# history = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1)
# print(model.evaluate(y_train, y_test))
#
# import matplotlib.pyplot as plt
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.legend(['training', 'valivation'], loc='upper left')
# plt.show()


# 3、GRU
def gru_model():
    model = keras.Sequential([
        layers.Embedding(input_dim=30000, output_dim=32, input_length=maxlen),
        layers.GRU(units=32, return_sequences=True),
        layers.GRU(units=1, activation='sigmoid', return_sequences=False)
    ])

    model.compile(optimizer=keras.optimizers.Adam(),
                  loss=keras.losses.BinaryCrossentropy(),
                  metrics=['accuracy'])
    return model

model = gru_model()
model.summary()

history = model.fit(x_train, y_train, batch_size=64, epochs=5,validation_split=0.1)
print(model.evaluate(y_train, y_test))
import matplotlib.pyplot as plt
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valivation'], loc='upper left')
plt.show()