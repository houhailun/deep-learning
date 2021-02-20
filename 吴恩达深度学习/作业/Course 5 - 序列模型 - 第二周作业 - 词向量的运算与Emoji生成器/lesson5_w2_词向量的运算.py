#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2021/2/10 10:29
# Author: Hou hailun

# 1 - 词向量运算
# 1.1 - 余弦相似度
# 1.2 - 词类类比
# 1.3 - 去除词向量中的偏见（选学）
# 1.3.1 - 消除与性别无关的词汇的偏差
# 1.3.2 - 性别词的均衡算法
# 2 - Emoji表情生成器
# 2.1 - 基准模型：Emojifier-V1
# 2.1.1 数据集
# 2.1.2 - Emojifier-V1的结构
# 2.1.3 - 实现Emojifier-V1
# 2.1.4 - 验证测试集
# 2.2 - Emojifier-V2：在Keras中使用LSTM模块
# 2.2.1 - 模型预览
# 2.2.2 - Keras与mini-batching
# 2.2.3 - 嵌入层（ The Embedding layer）
# 2.3 - 构建Emojifier-V2
# 3 - 博主推荐

#
# 1 - 词向量运算
#

# 因为词嵌入的训练是非常耗资源的，所以大部分人都是选择加载训练好的词嵌入数据。在本博客中，我们将学习到：
#   1、如何加载训练好了的词向量
#   2、使用余弦相似性计算相似度
#   3、使用词嵌入来解决“男人与女人相比就像国王与____ 相比”之类的词语类比问题
#   4、修改词嵌入以减少性别偏见等
import numpy as np
import w2v_utils

# 加载词向量
words, word_to_vec_map = w2v_utils.read_glove_vecs('data/glove.6B.50d.txt')
# words：单词的集合
# word_to_vec_map ： 字典类型，单词到GloVe向量的映射


# 1.1 - 余弦相似度
# 为了衡量两个词的相似程度，我们需要一种方法来衡量两个词的词嵌入向量之间的相似程度
def cosine_similarity(u, v):
    """
    u与v的余弦相似度反映了u与v的相似程度

    参数：
        u -- 维度为(n,)的词向量
        v -- 维度为(n,)的词向量

    返回：
        cosine_similarity -- 由上面公式定义的u和v之间的余弦相似度。
    """
    distince = 0

    # 计算u与v的内积
    dot = np.dot(u, v)

    # 计算u和v的L2范数
    norm_u = np.sqrt(np.sum(np.power(u, 2)))
    norm_v = np.sqrt(np.sum(np.power(v, 2)))

    # 计算余弦相似度
    cosine_similarity = np.divide(dot, norm_u * norm_v)

    return cosine_similarity


# 1.2 - 词类类比
# 学习解决“A与B相比就类似C与_相比”之类的问题
# 实际上我们需要找到一个词d，然后e_a, e_b, e_c, e_d 满足: e_d - e_a = e_d - e_b, 使用余弦相似性判断
def complete_analogy(word_a, word_b, word_c, word_to_vec_map):
    """
    解决“A与B相比就类似于C与____相比一样”之类的问题

    参数：
        word_a -- 一个字符串类型的词
        word_b -- 一个字符串类型的词
        word_c -- 一个字符串类型的词
        word_to_vec_map -- 字典类型，单词到GloVe向量的映射

    返回：
        best_word -- 满足(v_b - v_a) 最接近 (v_best_word - v_c) 的词
    """

    # 单词转换小写
    word_a, word_b, word_c = word_a.lower(), word_b.lower(), word_c.lower()

    # 获取对应单词的词向量
    e_a, e_b, e_c = word_to_vec_map[word_a], word_to_vec_map[word_b], word_to_vec_map[word_c]

    # 获取所有单词
    words = word_to_vec_map.keys()

    max_cosine_sim = -100
    best_word = None
    for word in words:
        # 避免匹配到输入数据
        if word in [word_a, word_b, word_c]:
            continue
        # 计算余弦相似度
        cosine_sim = cosine_similarity((e_b - e_a), (word_to_vec_map[word] - e_c))
        if cosine_sim > max_cosine_sim:
            max_cosine_sim = cosine_sim
            best_word = word

    return best_word


# triads_to_try = [('italy', 'italian', 'spain'), ('india', 'delhi', 'japan'), ('man', 'woman', 'boy'), ('small', 'smaller', 'large')]
# for triad in triads_to_try:
#     print('{} -> {} <====> {} -> {}'.format(*triad, complete_analogy(*triad, word_to_vec_map)))


# 1.3 去除词向量中的偏见
# 早期的硬Debias方法：
#   具体而言，它需要一组特定于性别的单词对，并计算这些单词对的差异向量的第一个主成分作为嵌入空间中的性别方向。
#   其次，它将有偏见的词嵌入投射到与推断的性别方向正交的子空间中，以消除性别偏见
#   问题：效果是有限的，因为性别偏见仍然可以从去偏后的嵌入几何图中恢复。

# 双重硬性偏差 Double-Hard Debias：通过消除频率影响来改善硬性偏差
#   关键思想是在应用Hard Debias之前将单词嵌入投影到intermediate subspace
#   首先将所有单词嵌入转换为「无频率子空间」，在该子空间中，我们能够计算出更准确的性别方向。
#       更具体地说，我们尝试找到对频率信息进行编码的维度，该频率信息分散了性别方向的计算。
#   然后，我们沿着字词嵌入的这个特定维度投影组件，以获得修正的嵌入，并对修正的嵌入应用Hard Debias。
#
# 详细步骤如下：
#   计算所有单词嵌入的主成分作为频率维度候选；
#   选择一组最偏（top-biased）的男性和女性词汇（例如，程序员，家庭主妇，游戏，舞蹈等）；
#   对没有候选维度 u_i 分别重读步骤4-6；
#   投影嵌入(embedding)到与 u_i 正交的中间空间中，从而获得经过修正的嵌入；
#   对修正的嵌入应用 Hard Debias；
#   对选定的top biased词的debiased embedding进行聚类，并计算聚类精度。

# 1.3.1 消除与性别无关的词汇的偏差
def neutralize(word, g, word_to_vec_map):
    """
    通过将“word”投影到与偏置轴正交的空间上，消除了“word”的偏差。
    该函数确保“word”在性别的子空间中的值为0

    参数：
        word -- 待消除偏差的字符串
        g -- 维度为(50,)，对应于偏置轴（如性别）
        word_to_vec_map -- 字典类型，单词到GloVe向量的映射

    返回：
        e_debiased -- 消除了偏差的向量。
    """

    # 根据word选择对应的词向量
    e = word_to_vec_map[word]

    # 计算词向量e在偏差g方向上的投影
    e_biascomponent = np.divide(np.dot(e, g), np.square(np.linalg.norm(9))) * g

    # 去除偏差
    e_debiased = e - e_biascomponent

    return e_debiased

g = word_to_vec_map['woman'] - word_to_vec_map['man']
# e = "receptionist"
# print("去偏差前{0}与g的余弦相似度为：{1}".format(e, cosine_similarity(word_to_vec_map["receptionist"], g)))
#
# e_debiased = neutralize("receptionist", g, word_to_vec_map)
# print("去偏差后{0}与g的余弦相似度为：{1}".format(e, cosine_similarity(e_debiased, g)))
# # NOTE：与博客中结果差别很大


# 1.3.2 性别词的均衡算法
# 关键思想是确保一对特定的单词与49维的g⊥距离相等
def equalize(pair, bias_axis, word_to_vec_map):
    """
    通过遵循上图中所描述的均衡方法来消除性别偏差。

    参数：
        pair -- 要消除性别偏差的词组，比如 ("actress", "actor")
        bias_axis -- 维度为(50,)，对应于偏置轴（如性别）
        word_to_vec_map -- 字典类型，单词到GloVe向量的映射

    返回：
        e_1 -- 第一个词的词向量
        e_2 -- 第二个词的词向量
    """
    # step1：获取词向量
    w1, w2 = pair
    e_w1, e_w2 = word_to_vec_map[w1], word_to_vec_map[w2]

    # step2: 计算w1与w2的均值
    mu = (e_w1 + e_w2) / 2.0

    # step3: 计算mu在偏差轴上的投影
    mu_B = np.divide(np.dot(mu, bias_axis), np.square(np.linalg.norm(bias_axis))) * bias_axis
    mu_other = mu - mu_B

    # step4: 计算e_w1B, e_w2B
    e_w1B = np.divide(np.dot(e_w1, bias_axis), np.square(np.linalg.norm(bias_axis))) * bias_axis
    e_w2B = np.divide(np.dot(e_w2, bias_axis), np.square(np.linalg.norm(bias_axis))) * bias_axis

    # step5: 调整e_w1B, e_w2B的偏差
    corrected_e_w1B = np.sqrt(np.abs(1 - np.square(np.linalg.norm(mu_other)))) * np.divide(e_w1B - mu_B, np.abs(e_w1 - mu_other - mu_B))
    corrected_e_w2B = np.sqrt(np.abs(1 - np.square(np.linalg.norm(mu_other)))) * np.divide(e_w2B - mu_B, np.abs(e_w2 - mu_other - mu_B))

    # step6: 使e1和e2等于它们修正后的投影之和，从而消除偏差
    e1 = corrected_e_w1B + mu_other
    e2 = corrected_e_w2B + mu_other

    return e1, e2


# print("==========均衡校正前==========")
# print("cosine_similarity(word_to_vec_map[\"man\"], gender) = ", cosine_similarity(word_to_vec_map["man"], g))
# print("cosine_similarity(word_to_vec_map[\"woman\"], gender) = ", cosine_similarity(word_to_vec_map["woman"], g))
# e1, e2 = equalize(("man", "woman"), g, word_to_vec_map)
# print("\n==========均衡校正后==========")
# print("cosine_similarity(e1, gender) = ", cosine_similarity(e1, g))
# print("cosine_similarity(e2, gender) = ", cosine_similarity(e2, g))
