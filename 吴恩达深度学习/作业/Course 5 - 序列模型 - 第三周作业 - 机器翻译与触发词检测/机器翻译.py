#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2021/2/20 13:27
# Author: Hou hailun


# 作业:构建一个神经机器翻译 (NMT) 模型
# 描述: 将人类可读日期 (“25th of June, 2009”) 翻译为机器可读日期 (“2009-06-25”).
#       您将使用注意模型执行此操作, 序列模型中最复杂的序列之一。

# 1 机器翻译
#   1.1 - 将人类可读日期翻译成机器可读日期
#       1.1.1 - 数据集
#   1.2 - 带注意力的神经机器翻译
#       1.2.1 - 注意机制
#   1.3 - 可视化注意力 (选学)
#       1.3.1 - 从网络获取激活

import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Concatenate, Permute, Dot, Input, LSTM, Multiply
from tensorflow.keras.layers import RepeatVector, Dense, Activation, Lambda
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import load_model, Model
import tensorflow.keras.backend as K
import numpy as np

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date
from nmt_utils import *
import matplotlib.pyplot as plt


# 1.1 将人类可读日期翻译成机器可读日期
# 1.1.1 数据集
m = 10000
dataset, human_vocab, machine_vocab, inv_machine_vocab = load_dataset(m)
print(dataset[:10])
# dataset: 一个元组列表 (人类可读日期, 机器可读日期)。
# human_vocab: 一个python字典，将人类可读日期中使用的所有字符映射到整数值索引。
# machine_vocab: 一个python字典，将机器可读日期中使用的所有字符映射到整数值索引。这些索引不一定与 human_vocab 的索引一致。
# inv_machine_vocab: machine_vocab的逆字典，从索引到字符的映射。

Tx = 30  # 假设人类可读日期的最大长度; 如果我们得到更长的输入，我们将截断它
Ty = 10  # YYYY-MM-DD
X, Y, Xoh, Yoh = preprocess_data(dataset, human_vocab, machine_vocab, Tx, Ty)
print("X.shape:", X.shape)
print("Y.shape:", Y.shape)
print("Xoh.shape:", Xoh.shape)
print("Yoh.shape:", Yoh.shape)
# 说明：
#   X: 训练集中人类可读日期的处理版本, 其中每个字符都被它在 human_vocab 中映射该字符的索引替换。每个日期都使用特殊字符（）进一步填充。维度为 X.shape = (m, Tx)
#   Y: 训练集中机器可读日期的处理版本, 其中每个字符都被它在machine_vocab中映射的索引替换。 维度为 Y.shape = (m, Ty)。
#   Xoh: X 的 one-hot 版本, one-hot 中条目 “1” 的索引被映射到在human_vocab中对应字符。维度为 Xoh.shape = (m, Tx, len(human_vocab))
#   Yoh: Y 的 one-hot 版本, one-hot 中条目 “1” 的索引被映射到由于machine_vocab 中对应字符。维度为 Yoh.shape = (m, Tx, len(machine_vocab))。 这里, len(machine_vocab) = 11 因为有 11 字符 (’-’ 以及 0-9).
index = 0
print("Source date:", dataset[index][0])
print("Target date:", dataset[index][1])
print()
print("Source after preprocessing (indices):", X[index])
print("Target after preprocessing (indices):", Y[index])
print()
print("Source after preprocessing (one-hot):", Xoh[index])
print("Target after preprocessing (one-hot):", Yoh[index])

# 1.2 带注意力的神经机器翻译
# 1.2.1 注意机制

# 将共享层定义为全局变量
repeator = RepeatVector(Tx)  # 将数据重复Tx词
concatenator = Concatenate(axis=-1)
densor1 = Dense(10, activation = "tanh")
densor2 = Dense(1, activation = "relu")
activator = Activation('softmax', name='attention_weights') # 在这个 notebook 我们正在使用自定义的 softmax(axis = 1)
dotor = Dot(axes=1)


def one_step_attention(a, s_prev):
    """
    执行一步 attention: 输出一个上下文向量，输出作为注意力权重的点积计算的上下文向量
    "alphas"  Bi-LSTM的 隐藏状态 "a"

    参数：
    a --  Bi-LSTM的输出隐藏状态 numpy-array 维度 (m, Tx, 2*n_a)
    s_prev -- (post-attention) LSTM的前一个隐藏状态, numpy-array 维度(m, n_s)

    返回：
    context -- 上下文向量, 下一个(post-attetion) LSTM 单元的输入
    :param a:
    :param s_prev:
    :return:
    """

    # 使用 repeator 重复 s_prev 维度 (m, Tx, n_s) 这样你就可以将它与所有隐藏状态"a" 连接起来。 (≈ 1 line)
    s_prev = repeator(s_prev)
    # 使用 concatenator 在最后一个轴上连接 a 和 s_prev (≈ 1 line)
    concat = concatenator([a, s_prev])
    # 使用 densor1 传入参数 concat, 通过一个小的全连接神经网络来计算“中间能量”变量 e。(≈1 lines)
    e = densor1(concat)
    # 使用 densor2 传入参数 e , 通过一个小的全连接神经网络来计算“能量”变量 energies。(≈1 lines)
    energies = densor2(e)
    # 使用 activator 传入参数 "energies" 计算注意力权重 "alphas" (≈ 1 line)
    alphas = activator(energies)
    # 使用 dotor 传入参数 "alphas" 和 "a" 计算下一个（(post-attention) LSTM 单元的上下文向量 (≈ 1 line)
    context = dotor([alphas, a])

    return context

n_a = 32
n_s = 64
post_activation_LSTM_cell = LSTM(n_s, return_state=True)
output_layer = Dense(len(machine_vocab), activation='softmax')


def model(Tx, Ty, n_a, n_s, human_vocab_size, machine_vocab_size):
    """
    参数:
        Tx -- 输入序列的长度
        Ty -- 输出序列的长度
        n_a -- Bi-LSTM的隐藏状态大小
        n_s -- post-attention LSTM的隐藏状态大小
        human_vocab_size -- python字典 "human_vocab" 的大小
        machine_vocab_size -- python字典 "machine_vocab" 的大小

    返回：
        model -- Keras 模型实例
    """

    # 定义模型的输入，维度(Tx,)
    # 定义s0, c0, 初始化解码器LSTM的隐藏状态，维度 (n_s,)
    X = Input(shape=(Tx, human_vocab_size))
    s0 = Input(shape=(n_s,), name='s0')
    c0 = Input(shape=(n_s,), name='c0')
    s = s0
    c = c0

    # 初始化一个空的输出列表
    outputs = []

    # step1: 定义 pre-attention Bi-LSTM。 记得使用 return_sequences=True. (≈ 1 line)
    a = Bidirectional(layer=LSTM(units=n_a, return_sequences=True),
                      input_shape=(m, Tx, n_a * 2))(X)

    # step2：迭代Ty步
    for t in range(Ty):
        # step2-1：执行一步注意力机制，得到在t步的上下文向量
        context = one_step_attention(a, s)

        # step2-2: 使用post-attention LSTM 单元得到新的 "context"
        # 别忘了使用： initial_state = [hidden state, cell state] (≈ 1 line)
        s, _, c = post_activation_LSTM_cell(inputs=context, initial_state=[s, c])

        # step2-3: 使用全连接层处理post-attention LSTM 的隐藏状态输出 (≈ 1 line)
        out = output_layer(s)

        # step2-4: 追加 "out" 到 "outputs" 列表 (≈ 1 line)
        outputs.append(out)

    # step3: 创建模型实例，获取三个输入并返回输出列表。 (≈ 1 line)
    model = Model(inputs=[X, s0, c0], outputs=outputs)

    return model


model = model(Tx, Ty, n_a, n_s, len(human_vocab), len(machine_vocab))
model.summary()

opt = Adam(lr=0.005, beta_1=0.9, beta_2=0.999, decay=0.01)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])


s0 = np.zeros((m, n_s))
c0 = np.zeros((m, n_s))
outputs = list(Yoh.swapaxes(0, 1))
model.fit([Xoh, s0, c0], outputs, epochs=1, batch_size=100)