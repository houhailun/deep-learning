#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/12/15 14:53
# Author: Hou hailun

# Seq2Seq for English - Arabic Translation in Keras
# 英语翻译为阿拉伯语

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


import re
import matplotlib.pyplot as plt

import os
path = './datasets'
print(os.listdir(path))

#
# 读取语料库
#
df_raw = pd.read_csv(f'{path}/fra.txt', delimiter='\t', error_bad_lines=False,
                     header=None, names=['en', 'fr'])
print(df_raw.head())

lang1 = 'en'
lang2 = 'fr'
# 英文最大长度为250
# 阿拉伯语最大长度为260
print(df_raw.count()[0], df_raw[lang1].str.len().max(), df_raw[lang2].str.len().max())
# 160538 286 349

#
# 预处理数据
#
# 英语字母
english_chars = [char for char in 'abcdefghijklmnopqrstuvwxyz1234567890']
english_chars_num = len(english_chars)
english_chars_stop = np.zeros(shape=(english_chars_num))
english_cleanup_regex = '[?!\.,/]'

# 阿拉伯语字符
arabic_chars = [char for char in '1234567890ابتثجحخدذرزسشصضطظعغفقكلمنهويءآٱأإةؤئى']
arabic_chars_num = len(arabic_chars)
arabic_chars_stop = np.zeros(shape=(arabic_chars_num))
arabic_cleanup_regex = '[ـ،؛؟٫٬٠]'

# 法语 characters
french_chars = [char for char in 'abcdefghijklmnopqrstuvwxyz1234567890']
french_chars_num = len(english_chars)
french_char_stop = np.zeros((english_chars_num))
french_cleanup_regex = '[?!\.,/]'

# parameters for the source and destination languages
params = {
    'en': {
        'chars': english_chars,
        'chars_num': english_chars_num,
        'char_stop': english_chars_stop,
        'cleanup_regex': english_cleanup_regex,
        'max_sentence_length': 70
    },
    'fr': {
        'chars': french_chars,
        'chars_num': french_chars_num,
        'char_stop': french_char_stop,
        'cleanup_regex': french_cleanup_regex,
        'max_sentence_length': 70
    }
}


# transform a sentence into an matrix of dimension (max_sentence_en, num_english_characters)
def one_hot_encoding(encoding, sentence, lang):
    # 使用给定的字符集和encoding长度来做one-hot
    if lang not in params.keys():
        print('unknown lang:', lang)
        return
    p = params[lang]
    chars, length, max_len, stop_char = p['chars'],\
                                        p['chars_num'],\
                                        p['max_sentence_length'],\
                                        p['char_stop']
    # 对语句sentence中每个字符做one-hot, 构造二维数组
    for index in range(len(sentence)):
        char = sentence[index]
        if char in chars:  # 假设字典中只有1个char
            idx = chars.index(char)
            onehot = np.zeros(shape=(length))
            onehot[idx] = 1
            encoding[index] = onehot
    # padding with stop char
    padding = max_len - len(sentence)
    for i in range(padding):
        encoding[len(sentence) + i] = stop_char


# TODO: 为什么要右移？
# shift the given encoding array to the right
def shift_right(encodings, lang):
    if lang not in params.keys():
        print('unknown lang:', lang)
        return
    p = params[lang]
    target = np.zeros(shape=(num_pairs, p['max_sentence_length'], p['chars_num']))
    stop = p['char_stop']
    # shift
    rows, sentences, chars = encodings.shape
    for r in range(rows):
        for s in range(sentences-1):
            target[r][s] = encodings[r][s+1]
        target[r][sentences-1] = stop
    return target

languages = params.keys()
for lang in languages:
    df_raw[lang] = df_raw[lang].str.lower()
    df_raw[lang] = df_raw[lang].str.strip()
    # 正则化，去除停用词
    df_raw[lang] = df_raw[lang].map(lambda x: re.sub(params[lang]['cleanup_regex'], '', x))
    # 过滤超出长度范围的句子
    df_raw = df_raw[(df_raw[lang].str.len() <= params[lang]['max_sentence_length'])]

num_pairs = df_raw.count()[0]  # 句子个数


# 分析语句长度
# 语句不同长度的个数
def plot_sentence_dist(df_raw, lang):
    df_agg = pd.DataFrame(df_raw[lang].map(lambda x: len(x)))
    df_agg[lang+'_count'] = 1
    return df_agg.groupby(lang).agg('count')


df_agg1 = plot_sentence_dist(df_raw, lang1)
df_agg2 = plot_sentence_dist(df_raw, lang2)
# print(df_agg1[lang1+'_count'].describe())
# print(df_agg2[lang2+'_count'].describe())

plt.subplot(1, 2, 1)
plt.plot(df_agg1)
plt.title('sentences length')
plt.ylabel('count')
plt.xlabel(lang1)

plt.subplot(1, 2, 2)
plt.plot(df_agg2)
plt.title('sentences length')
plt.xlabel(lang2)
# 分布基本一致

#df_agg1[lang1+'_count']
plt.figure(figsize=(15, 5))
plt.plot(df_agg1[lang1+'_count'].keys(), df_agg1[lang1+'_count'].values, marker='o', label=lang1)
plt.plot(df_agg2[lang2+'_count'].keys(), df_agg2[lang2+'_count'].values, marker='o', label=lang2)
plt.legend()
plt.xlabel('Length of sentence')
plt.ylabel('Count of occurrences')
# plt.tight_layout()
# plt.show()


#
# seq2seq输入
#
encodings = {
    'fr': np.zeros(shape=(num_pairs, params['fr']['max_sentence_length'], params['fr']['chars_num'])),
    'en': np.zeros(shape=(num_pairs, params['en']['max_sentence_length'], params['en']['chars_num']))
}

index = 0
for row_idx, row in df_raw.iterrows():  # 按行迭代
    for lang in encodings.keys():  # lang: fr, en
        one_hot_encoding(encodings[lang][index], row[lang], lang)
    index += 1
print(encodings['en'].shape, encodings['fr'].shape)
# (156020, 70, 36) (156020, 70, 36)  156020个语句，每个语句有70个单词，每个单词是36个中的一个(one-hot)

# A 3D array of shape (num_pairs, max_sentence_en, num_english_characters) containing a one-hot vectorization of the English sentences.
encoder_input_data = encodings['en']

# A 3D array of shape (num_pairs, max_sentence_ar, num_arabic_characters) containg a one-hot vectorization of the Arabic sentences.
decoder_input_data = encodings['fr']

# Same as decoder_input_data but offset by one timestep. decoder_target_data[:, t, :] will be the same as decoder_input_data[:, t + 1, :].
decoder_target_data = shift_right(encodings['fr'], 'fr')
print(encoder_input_data.shape, decoder_input_data.shape, decoder_target_data.shape)


#
# LSTM-based Architecture
#
# language model
max_token_lang1 = params['en']['chars_num']
max_sentence_length_lang1 = params['en']['max_sentence_length']
max_token_lang2 = params['fr']['chars_num']
max_sentence_length_lang2 = params['fr']['max_sentence_length']

batch_size = 64
epochs = 10
latent_dim = 256  # Latent dimensionality of the encoding space

# initializer the shape of the input for the Encoder/Decoder
encoder_input = layers.Input(shape=(None, max_token_lang1))
decoder_input = layers.Input(shape=(None, max_token_lang2))

# Encoding-Decoding lSTM
encoder = layers.LSTM(units=latent_dim, return_state=True)
decoder = layers.LSTM(units=latent_dim, return_state=True, return_sequences=True)

# run english input sentence througth the encoder
encoder_input_reshaped = layers.Reshape((max_sentence_length_lang1, max_token_lang1))(encoder_input)
_, h, c = encoder(encoder_input_reshaped)

# pass the hidden/context states from encoder to the decoder, along with the target Arabic sentence
encoder_input_reshape = layers.Reshape((max_sentence_length_lang2, max_token_lang2))(decoder_input)
out, _, _ = decoder(encoder_input_reshaped, initial_state=[h, c])

# generate the final output
out = layers.Dense(max_token_lang2)(out)
decoder_output = layers.Activation('softmax')(out)

# create the keras model
model = keras.Model(inputs=[encoder_input, decoder_input], outputs=[decoder_output])

adam = keras.optimizers.Adam()
model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])

model.summary()

samples = decoder_input_data.shape[0]

model.fit(x=[encoder_input_data[:samples], decoder_input_data[:samples]],
          y=[decoder_target_data[:samples]],
          batch_size=batch_size,
          shuffle=True,
          epochs=epochs,
          validation_split=0.1)


#
# 模型存储
#
def store(model):
    model_json = model.to_json()
    with open('model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('model.h5')
    print('saved model to disk')


def restore():
    # load json and create model
    json_file = open('model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = keras.models.model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("model.h5")
    print("Loaded model from disk")
    return loaded_model


store(model)

print(df_agg1[lang1+'_count'].keys())