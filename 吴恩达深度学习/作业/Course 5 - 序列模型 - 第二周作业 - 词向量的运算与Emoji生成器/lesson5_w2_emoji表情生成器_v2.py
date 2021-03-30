#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2021/2/19 18:11
# Author: Hou hailun

# 使用keras lstm模型构建表情生成器
import numpy as np
np.random.seed(0)
import tensorflow as tf
import tensorflow.keras as keras
from  tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Dropout, LSTM, Activation
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing import sequence

np.random.seed(1)
from tensorflow.keras.initializers import glorot_uniform

import emo_utils


X_train, Y_train = emo_utils.read_csv('data/train_emoji.csv')
X_test, Y_test = emo_utils.read_csv('data/test.csv')
word_to_index, index_to_word, word_to_vec_map = emo_utils.read_glove_vecs('data/glove.6B.50d.txt')

# 2.2.1 模型概览
# 模型描述：采用两层的LSTM序列分类器

# 2.2.2 Keras与mini-batching
# 框架要求输入向量长度相同，对于不同长度的输入需进行填充
# 方法：找到最长句子，获取长度，填充到该长度


# 2.2.3 嵌入层
def sentences_to_indices(X, word_to_index, max_len):
    """
    输入的是X（字符串类型的句子的数组），再转化为对应的句子列表，
    输出的是能够让Embedding()函数接受的列表或矩阵（参见图4）。

    参数：
        X -- 句子数组，维度为(m, 1)
        word_to_index -- 字典类型的单词到索引的映射
        max_len -- 最大句子的长度，数据集中所有的句子的长度都不会超过它。

    返回：
        X_indices -- 对应于X中的单词索引数组，维度为(m, max_len)
    """
    m = X.shape[0]  # 训练集数量
    X_indices = np.zeros(shape=(m, max_len))  # 0初始化词索引数组

    for i in range(m):
        # 将第i个样本转换小写并按单词分开
        sentences_words = X[i].lower().split()

        j = 0
        # 遍历单词列表
        for w in sentences_words:
            # 将X_indices的第(i, j)号元素为对应的单词索引
            X_indices[i, j] = word_to_index[w]
            j += 1

    return X_indices


# X1 = np.array(["funny lol", "lets play baseball", "food is ready for you"])
# X1_indices = sentences_to_indices(X1, word_to_index, max_len=5)
# print("X1 =", X1)
# print("X1_indices =", X1_indices)


# 构建词向量
def pretrained_embedding_layer(word_to_vec_map, word_to_index):
    """
    创建Keras Embedding()层，加载已经训练好了的50维GloVe向量

    参数：
        word_to_vec_map -- 字典类型的单词与词嵌入的映射
        word_to_index -- 字典类型的单词到词汇表（400,001个单词）的索引的映射。

    返回：
        embedding_layer() -- 训练好了的Keras的实体层。
    """
    vocab_len = len(word_to_index) + 1               # 词汇表大小
    emd_dim = word_to_vec_map['cucumber'].shape[0]   # embedding dim

    # 初始化嵌入矩阵
    emb_matrix = np.zeros(shape=(vocab_len, emd_dim))

    # 将嵌入矩阵的每行的“index”设置为词汇“index”的词向量表示
    for word, index in word_to_index.items():
        emb_matrix[index, :] = word_to_vec_map[word]

    # 定义keras的embedding层
    embedding_layer = Embedding(input_dim=vocab_len, output_dim=emd_dim, trainable=False)

    # 构建embedding层
    embedding_layer.build(input_shape=(None,))

    # 将嵌入层的权重设置为嵌入矩阵
    embedding_layer.set_weights(weights=[emb_matrix])

    return embedding_layer


embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)
# print("weights[0][1][3] =", embedding_layer.get_weights()[0][1][3])


# 2.3 构建Emojifier-V2
def Emojify_V2(input_shape, word_to_vec_map, word_to_index):
    """
    实现Emojify-V2模型的计算图

    参数：
        input_shape -- 输入的维度，通常是(max_len,)
        word_to_vec_map -- 字典类型的单词与词嵌入的映射。
        word_to_index -- 字典类型的单词到词汇表（400,001个单词）的索引的映射。

    返回：
        model -- Keras模型实体
    """
    # 定义sentence_indices为计算图的输入，维度为(input_shape,)，类型为dtype 'int32'
    sentence_indices = Input(input_shape, dtype='int32')

    # 创建embedding层
    embedding_layer = pretrained_embedding_layer(word_to_vec_map, word_to_index)

    # 通过嵌入层传播sentence_indices，你会得到嵌入的结果
    embeddings = embedding_layer(sentence_indices)

    # 通过带有128维隐藏状态的LSTM层传播嵌入
    # 需要注意的是，返回的输出应该是一批序列。
    X = LSTM(units=128, return_sequences=True)(embeddings)  # return_sequences=True 返回所有hidden state值
    X = Dropout(0.5)(X)

    X = LSTM(units=128, return_sequences=False)(X)
    X = Dropout(0.5)(X)

    X = Dense(units=5)(X)
    X = Activation('softmax')(X)

    # 创建模型实体
    model = Model(inputs=sentence_indices, outputs=X)

    return model

maxLen=10
model = Emojify_V2((maxLen,), word_to_vec_map, word_to_index)
model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

X_train_indices = sentences_to_indices(X_train, word_to_index, maxLen)

Y_train_oh = emo_utils.convert_to_one_hot(Y_train, C=5)
model.fit(x=X_train_indices, y=Y_train_oh, epochs=50, batch_size=32, shuffle=True)


X_test_indices = sentences_to_indices(X_test, word_to_index, max_len = maxLen)
Y_test_oh = emo_utils.convert_to_one_hot(Y_test, C = 5)
loss, acc = model.evaluate(X_test_indices, Y_test_oh)

print("Test accuracy = ", acc)