#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/10/26 19:10
# Author: Hou hailun

# 使用tensorflow实现word2vec

import collections
import math
import numpy as np
import os
import random
import tensorflow as tf
import zipfile
from matplotlib import pylab
from six.moves import range
from six.moves.urllib.request import urlretrieve
from sklearn.manifold import TSNE  # word embedding的可视化算法，降维至2D

# ----------------- 下载数据 -----------------
url = 'http://mattmahoney.net/dc/'


def maybe_download(filename, expected_bytes):
    """Download a file if not present, and make sure it's the right size."""
    if not os.path.exists(filename):
        # urlretrieve(): 将远程数据下载到本地
        filename, _ = urlretrieve(url + filename, filename)
    statinfo = os.stat(filename)
    if statinfo.st_size == expected_bytes:
        print('Found and verified %s' % filename)
    else:
        print(statinfo.st_size)
        raise Exception('Failed to verify ' + filename + '. Can you get to it with a browser?')
        return filename

filename = maybe_download('text8.zip', 31344016)


# ----------------- 解析数据 -----------------
def read_data(filename):
    # 里面只有一个文件text8，包含了多个单词
    # f.read返回字节，tf.compat.as_str将字节转为字符
    # data包含了所有单词
    with zipfile.ZipFile(filename) as f:
        data = tf.compat.as_str(f.read(f.namelist()[0])).split()
    return data

words = read_data(filename)
print('Data size %d' % len(words))


# ----------------- 构建词汇表 -----------------
vocab_size = 50000  # 词汇表大小


def build_dataset(words):
    count = [['UNK', 1]]  # 表示未知，即不在词汇表里的单词，注意这里用的是列表形式而非元组形式，因为后面未知的数量需要赋值
    count.extend(collections.Counter(words).most_common(vocab_size-1))  # 出现次数最多的5w个单词及次数

    # 词-索引哈希
    # 根据词频从大到小对单词进行编码，tf中的Tokenizer类
    dictionary = dict()
    for word, _ in count:
        dictionary[word] = len(dictionary)  # 每增加一个-->len+1，索引从0开始

    # 用索引表示的整个text8文本
    data = list()
    unk_count = 0
    # 记录单词index
    # 在词汇表中的单词记录ix
    # 词汇表中未出现的单词在0位置记录UNK
    for word in words:
        if word in dictionary:
            index = dictionary[word]
        else:
            index = 0  # dictionary['UNK']
            unk_count += 1
        data.append(index)
    count[0][1] = unk_count
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))  # 索引-词哈希
    return data, count, dictionary, reverse_dictionary

data, count, dictionary, reverse_dictionary = build_dataset(words)
print('Most common words (+UNK)', count[:5])
print('Sample data', data[:10])
del words  # Hint to reduce memory.


# ----------------- 生成batch函数 -----------------
data_index = 0


def generate_batch(batch_size, num_skips, skip_window):
    """
    生成batch函数
    :param batch_size: 指定每次取训练集样本个数
    :param num_skips: 表示在两侧窗口内总共取多少次，数量小于2 * ship_window
    :param skip_window: 滑动窗口大小
    :return:
    """
    global data_index
    assert batch_size % num_skips == 0
    assert num_skips <= 2 * skip_window

    # 初始化batch和labels
    batch = np.ndarray(shape=(batch_size), dtype=np.int32)
    labels = np.ndarray(shape=(batch_size, 1), dtype=np.int32)

    span = 2 * skip_window + 1  # [ skip_window target skip_window ]
    # buffer这个队列太有用了，不断地保存span个单词在里面，然后不断往后滑动，而且buffer[skip_window]就是中心词
    buffer = collections.deque(maxlen=span)
    for _ in range(span):
        buffer.append(data[data_index])
        data_index = (data_index + 1) % len(data)

    # 需要多少个中心词
    # 因为一个target对应num_skips个的单词，即一个目标单词w在num_skips=2时形成2个样本(w,left_w),(w,right_w)
    # 这样描述了目标单词w的上下文
    center_words_count = batch_size // num_skips
    for i in range(center_words_count):
        # skip_window在buffer里正好是中心词所在位置
        target = skip_window
        targets_to_avoid = [skip_window]
        for j in range(num_skips):
            # 选取span窗口中不包含target的，且不包含已选过的
            target = random.choice([i for i in range(0, span) if i not in targets_to_avoid])
            targets_to_avoid.append(target)
            # batch中记录中心词
            batch[i * num_skips + j] = buffer[skip_window]
            # label记录上下文，同一个target对应num_skips个上下文单词
            labels[i * num_skips + j, 0] = buffer[target]
        buffer.append(data[data_index])  # buffer滑动一格
        data_index = (data_index + 1) % len(data)
    return batch, labels

# 打印前8个单词
print('data:', [reverse_dictionary[di] for di in data[:10]])
for num_skips, skip_window in [(2, 1), (4, 2)]:
    data_index = 0
    batch, labels = generate_batch(batch_size=16, num_skips=num_skips, skip_window=skip_window)
    print('\nwith num_skips = %d and skip_window = %d:' % (num_skips, skip_window))
    print('    batch:', [reverse_dictionary[bi] for bi in batch])
    print('    labels:', [reverse_dictionary[li] for li in labels.reshape(16)])


# ----------------- 构建model -----------------
batch_size = 128
embedding_size = 128
skip_window = 1  # 上下文只关注左右单词
num_skips = 2    # 抽2次上下文

valid_size = 16
valid_window = 100
# 随机挑选一组单词作为验证集，valid_examples也就是下面的valid_dataset，是一个一维的ndarray
valid_examples = np.array(random.sample(range(valid_window), valid_size))

# trick: 负采样数值
num_sampled = 64

graph = tf.Graph()

with graph.as_default():
    # 训练集和标签，以及验证集（注意验证集是一个常量集合）
    train_dataset = tf.placeholder(tf.int32, shape=[batch_size])
    train_labels = tf.placeholder(tf.int32, shape=[batch_size, 1])
    valid_dataset = tf.constant(valid_examples, dtype=tf.int32)

    # 定义embedding层，初始化
    # 创建嵌入变量（每一行代表一个词嵌入向量）
    embeddings = tf.Variable(tf.random_uniform_initializer([vocab_size, embedding_size], -1.0, 1.0))
    softmax_weights = tf.Variable(tf.truncated_normal([vocab_size, embedding_size], stddev=1.0 / math.sqrt(embedding_size)))
    softmax_biases = tf.Variable(tf.zeros([vocab_size]))

    # model
    # train_dataset通过embeddings变为稠密向量，train_dataset是一个一维的ndarray
    embed = tf.nn.embedding_lookup(embeddings, train_dataset)  # 对于X中的每一个样本查找对应的嵌入向量

    # Compute the softmax loss, using a sample of the negative labels each time.
    # 计算损失，tf.reduce_mean和tf.nn.sampled_softmax_loss
    loss = tf.reduce_mean(tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                                     labels=train_labels, num_sampled=num_sampled,
                                                     num_classes=vocab_size))

    # Optimizer.优化器，这里也会优化embeddings
    # Note: The optimizer will optimize the softmax_weights AND the embeddings.
    # This is because the embeddings are defined as a variable quantity and the
    # optimizer's `minimize` method will by default modify all variable quantities
    # that contribute to the tensor it is passed.
    # See docs on `tf.train.Optimizer.minimize()` for more details.
    optimizer = tf.train.AdagradOptimizer(1.0).minimize(loss)

    # 模型其实到这里就结束了，下面是在验证集上做效果验证
    # Compute the similarity between minibatch examples and all embeddings.
    # We use the cosine distance:先对embeddings做正则化
    norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
    normalized_embeddings = embeddings / norm
    valid_embeddings = tf.nn.embedding_lookup(normalized_embeddings, valid_dataset)
    # 验证集单词与其他所有单词的相似度计算
    similarity = tf.matmul(valid_embeddings, tf.transpose(normalized_embeddings))


# ------------------------------ 开始训练 ------------------------------
num_steps = 40001
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    print('Initialized')
    average_loss = 0
    for step in range(num_steps):
        batch_data, batch_labels = generate_batch(batch_size, num_skips, skip_window)
        feed_dict = {train_dataset: batch_data, train_labels: batch_labels}
        _, this_loss = session.run([optimizer, loss], feed_dict=feed_dict)

        average_loss += this_loss
        #     每2000步计算一次平均loss
        if step % 2000 == 0:
            if step > 0:
                average_loss = average_loss / 2000
            # The average loss is an estimate of the loss over the last 2000 batches.
            print('Average loss at step %d: %f' % (step, average_loss))
            average_loss = 0

        # note that this is expensive (~20% slowdown if computed every 500 steps)
        if step % 10000 == 0:
            sim = similarity.eval()
            for i in range(valid_size):
                valid_word = reverse_dictionary[valid_examples[i]]
                top_k = 8  # number of nearest neighbors
                #         nearest = (-sim[i, :]).argsort()[1:top_k+1]
                nearest = (-sim[i, :]).argsort()[0:top_k + 1]  # 包含自己试试
                log = 'Nearest to %s:' % valid_word
                for k in range(top_k):
                    close_word = reverse_dictionary[nearest[k]]
                    log = '%s %s,' % (log, close_word)
                print(log)
    # 一直到训练结束，再对所有embeddings做一次正则化，得到最后的embedding
    final_embeddings = normalized_embeddings.eval()


# ------------------------- 可视化 --------------------------
num_points = 400
# 降维度PCA
tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
two_d_embeddings = tsne.fit_transform(final_embeddings[1:num_points+1, :])
def plot(embeddings, labels):
  assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'
  pylab.figure(figsize=(15,15))  # in inches
  for i, label in enumerate(labels):
    x, y = embeddings[i,:]
    pylab.scatter(x, y)
    pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                   ha='right', va='bottom')
  pylab.show()

words = [reverse_dictionary[i] for i in range(1, num_points+1)]
plot(two_d_embeddings, words)