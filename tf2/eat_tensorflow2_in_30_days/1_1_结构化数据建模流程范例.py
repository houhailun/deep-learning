#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Time: 2020/9/8 13:33
# Author: Hou hailun

# https://github.com/lyhue1991/eat_tensorflow2_in_30_days

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import models
from tensorflow.keras import layers

dfTrain_raw = pd.read_csv('../datasets/titanic/train.csv')
dfTest_raw = pd.read_csv('../datasets/titanic/test.csv')

# 字段说明：
#
# Survived:0代表死亡，1代表存活【y标签】
# Pclass:乘客所持票类，有三种值(1,2,3) 【转换成onehot编码】
# Name:乘客姓名 【舍去】
# Sex:乘客性别 【转换成bool特征】
# Age:乘客年龄(有缺失) 【数值特征，添加“年龄是否缺失”作为辅助特征】
# SibSp:乘客兄弟姐妹/配偶的个数(整数值) 【数值特征】
# Parch:乘客父母/孩子的个数(整数值)【数值特征】
# Ticket:票号(字符串)【舍去】
# Fare:乘客所持票的价格(浮点数，0-500不等) 【数值特征】
# Cabin:乘客所在船舱(有缺失) 【添加“所在船舱是否缺失”作为辅助特征】
# Embarked:乘客登船港口:S、C、Q(有缺失)【转换成onehot编码，四维度 S,C,Q,nan】

ax = dfTrain_raw['Survived'].value_counts().plot(kind='bar')
plt.show()