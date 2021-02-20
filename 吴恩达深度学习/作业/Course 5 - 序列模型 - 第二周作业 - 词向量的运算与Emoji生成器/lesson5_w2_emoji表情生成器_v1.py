#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2021/2/19 16:26
# Author: Hou hailun

import numpy as np
import emo_utils
import emoji
import matplotlib.pyplot as plt

# 2.1 基准模型: Emojifier_V1
# 2.1.1 数据集
# (X, Y)  X：包含127个字符串的短句  Y: 包含了对应短句的标签(0~4)
X_train, Y_train = emo_utils.read_csv('data/train_emoji.csv')
X_test, Y_test = emo_utils.read_csv('data/test.csv')

# maxLen = len(max(X_train, key=len).split())
# print(maxLen)  # 10

# index = 3
# print(X_train[index], emo_utils.label_to_emoji(Y_train[index]))


# 2.1.2 Emojifier-V1的结构
#   输入：一段文字，比如 I loce you
#   输出的是维度为(1,5)的向量，
#   然后再argmax层寻找最大可能性的输出
Y_oh_train = emo_utils.convert_to_one_hot(Y_train, C=5)  # one-hot （m，1）->（m, 5）
Y_oh_test = emo_utils.convert_to_one_hot(Y_test, C=5)

# index = 0
# print("{0}对应的独热编码是{1}".format(Y_train[index], Y_oh_train[index]))

# 2.1.3 实现Emojifier_V1
# step1: 把输入的句子转换为词向量，然后获取均值
word_to_index, index_to_word, word_to_vec_map = emo_utils.read_glove_vecs('data/glove.6B.50d.txt')
# word_to_index：字典类型的词汇（400,001个）与索引的映射（有效范围：0-400,000）
# index_to_word：字典类型的索引与词汇之间的映射。
# word_to_vec_map：字典类型的词汇与对应GloVe向量的映射。

# word = "cucumber"
# index = 113317
# print("单词{0}对应的索引是：{1}".format(word, word_to_index[word]))
# print("索引{0}对应的单词是：{1}".format(index, index_to_word[index]))


def sentence_to_avg(sentence, word_to_vec_map):
    """
    将句子转换为单词列表，提取其GloVe向量，然后将其平均。

    参数：
        sentence -- 字符串类型，从X中获取的样本。
        word_to_vec_map -- 字典类型，单词映射到50维的向量的字典

    返回：
        avg -- 对句子的均值编码，维度为(50,)
    """
    # 第一步: 分割句子，转换为列表
    words = sentence.lower().split()

    # 初始化均值词向量
    avg = np.zeros(shape=(50,))  # 50行1列

    # 第二步: 对词向量取平均
    for w in words:
        avg += word_to_vec_map[w]
    avg = np.divide(avg, len(words))

    return avg


# avg = sentence_to_avg("Morrocan couscous is my favorite dish", word_to_vec_map)
# print("avg = ", avg)


# 定义model
def model(X, Y, word_to_vec_map, learning_rate=0.01, num_iterations=400):
    """
    在numpy中训练词向量模型。

    参数：
        X -- 输入的字符串类型的数据，维度为(m, 1)。
        Y -- 对应的标签，0-7的数组，维度为(m, 1)。
        word_to_vec_map -- 字典类型的单词到50维词向量的映射。
        learning_rate -- 学习率.
        num_iterations -- 迭代次数。

    返回：
        pred -- 预测的向量，维度为(m, 1)。
        W -- 权重参数，维度为(n_y, n_h)。
        b -- 偏置参数，维度为(n_y,)
    """
    np.random.seed(1)

    m = Y.shape[0]  # 定义训练数量
    n_y = 5         # 类别维度
    n_h = 50        # 隐藏层维度

    # 使用Xavier初始化参数
    W = np.random.randn(n_y, n_h) / np.sqrt(n_h)
    b = np.zeros((n_y,))

    # 将Y转换未one-hot
    Y_oh = emo_utils.convert_to_one_hot(Y, C=n_y)

    # 循环优化
    for t in range(num_iterations):
        for i in range(m):
            # 获取第i个样本的均值
            avg = sentence_to_avg(X[i], word_to_vec_map)

            # 前向传播
            z = np.dot(W, avg) + b
            a = emo_utils.softmax(z)

            # 计算第i个样本的损失
            cost = -np.sum(Y_oh[i] * np.log(a))

            # 计算梯度
            dz = a - Y_oh[i]
            dW = np.dot(dz.reshape(n_y, 1), avg.reshape(1, n_h))
            db = dz

            # 更新参数
            W = W - learning_rate * dW
            b = b - learning_rate * db
        if t % 100 == 0:
            print("第{t}轮，损失为{cost}".format(t=t, cost=cost))
            pred = emo_utils.predict(X, Y, W, b, word_to_vec_map)

    return pred, W, b


print(X_train.shape)
print(Y_train.shape)
print(np.eye(5)[Y_train.reshape(-1)].shape)
print(X_train[0])
print(type(X_train))
Y = np.asarray([5,0,0,5, 4, 4, 4, 6, 6, 4, 1, 1, 5, 6, 6, 3, 6, 3, 4, 4])
print(Y.shape)

X = np.asarray(['I am going to the bar tonight', 'I love you', 'miss you my dear',
 'Lets go party and drinks','Congrats on the new job','Congratulations',
 'I am so happy for you', 'Why are you feeling bad', 'What is wrong with you',
 'You totally deserve this prize', 'Let us go play football',
 'Are you down for football this afternoon', 'Work hard play harder',
 'It is suprising how people can be dumb sometimes',
 'I am very disappointed','It is the best day in my life',
 'I think I will end up alone','My life is so boring','Good job',
 'Great so awesome'])

pred, W, b = model(X_train, Y_train, word_to_vec_map)

# 2.1.4 验证测试集
print("=====训练集====")
pred_train = emo_utils.predict(X_train, Y_train, W, b, word_to_vec_map)
print("=====测试集====")
pred_test = emo_utils.predict(X_test, Y_test, W, b, word_to_vec_map)

X_my_sentences = np.array(["i adore you", "i love you", "funny lol", "lets play with a ball", "food is ready", "you are not happy"])
Y_my_labels = np.array([[0], [0], [2], [1], [4],[3]])

pred = emo_utils.predict(X_my_sentences, Y_my_labels , W, b, word_to_vec_map)
emo_utils.print_predictions(X_my_sentences, pred)

print(" \t {0} \t {1} \t {2} \t {3} \t {4}".format(emo_utils.label_to_emoji(0), emo_utils.label_to_emoji(1), \
                                                 emo_utils.label_to_emoji(2), emo_utils.label_to_emoji(3), \
                                                 emo_utils.label_to_emoji(4)))
import pandas as pd
print(pd.crosstab(Y_test, pred_test.reshape(56,), rownames=['Actual'], colnames=['Predicted'], margins=True))
emo_utils.plot_confusion_matrix(Y_test, pred_test)