#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/10/28 14:17
# Author: Hou hailun

# 基于Gensim的word2vec模式使用
# 该数据集包括诸如汽车的品牌，型号，年份，引擎和其他属性的特征。
# 我们将使用这些功能为每个品牌模型生成word embedding，然后比较不同品牌模型之间的相似性。
# word2vec要求使用“list of lists”格式进行训练
# 具体地说，所有的品牌模型都包含在一个总的列表(lists)中，每个列表(list)都包含一个品牌的描述词。

import pandas as pd
from gensim.models import Word2Vec
import numpy as np
from sklearn import manifold
import matplotlib.pyplot as plt


# 数据读取
df = pd.read_csv('datasets/data.csv')

# 把 make 和 model 列合并
df['Maker_Model'] = df['Make'] + " " + df['Model']

# 选取需要训练的特征，主要为文本类特征
df1 = df[['Engine Fuel Type','Transmission Type','Driven_Wheels','Market Category','Vehicle Size', 'Vehicle Style', 'Maker_Model']]

# 将多列特征合并成一列
df2 = df1.apply(lambda x: ','.join(x.astype(str)), axis=1)  # 对每行中的多列特征组合
df_clean = pd.DataFrame({'clean': df2})
print(df_clean.head())

# 生成训练lists
sent = [row.split(',') for row in df_clean['clean']]
print(sent[:2])

# 训练word2vec模型
model = Word2Vec(sent, min_count=1, size=50, workers=3, window=3, sg=1)
print(model['Toyota Camry'])
print(len(model['Toyota Camry']))

# 计算相似性
print(model.similarity('Porsche 718 Cayman', 'Nissan Van'))   # 0.8510145
print(model.most_similar('Mercedes-Benz SLK-Class')[:5])      # 最相似


# 使用TSNE进行可视化
def display_closestwords_tsnescatterplot(model, word, size):
    arr = np.empty((0,size), dtype='f')
    word_labels = [word]
    close_words = model.similar_by_word(word)
    arr = np.append(arr, np.array([model[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = model[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)
    tsne = manifold.TSNE(n_components=2, random_state=0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
        plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points')
    plt.xlim(x_coords.min()+0.00005, x_coords.max()+0.00005)
    plt.ylim(y_coords.min()+0.00005, y_coords.max()+0.00005)
    plt.show()


display_closestwords_tsnescatterplot(model, 'Porsche 718 Cayman', 50)
