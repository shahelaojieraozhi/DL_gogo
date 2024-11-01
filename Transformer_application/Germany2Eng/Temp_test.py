# -*- coding: utf-8 -*-
"""
@Project ：DL_gogo 
@Time    : 2023/7/18 21:37
@Author  : Rao Zhi
@File    : Temp_test.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import numpy as np
import pandas as pd
# 导入 math 包
import math

# 输出一个数字的自然对数


# print(math.log(2.7183))
# print(math.log(2))
# print(math.log(1))

# df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c'], 'C': [2, 2, 3]})
# df = pd.DataFrame({'A': ['a', 'b', 'a'], 'B': ['b', 'a', 'c']})
# print(df)
# print(pd.get_dummies(df))
# print(pd.get_dummies(df, prefix=['col1', 'col2']))


# s = ['a', 'b', np.nan]
# print(pd.get_dummies(s))
# print(pd.get_dummies(s, dummy_na=True))

# s = ['a', 'b']
# print(pd.get_dummies(s))
# print(pd.get_dummies(s, dtype=float))

import torch
import torch.nn as nn

# embedding = nn.Embedding(10, 3)
# input = torch.LongTensor([[0, 2, 0, 5]])
# print(embedding(input))


embedding = nn.Embedding(3, 4)  # 假定字典中只有5个词，词向量维度为4
word = [[1, 2, 3]]  # 每个数字代表一个词，例如 {'!':0,'how':1, 'are':2, 'you':3,  'ok':4}
# 而且这些数字的范围只能在0～4之间，因为上面定义了只有5个词
embed = embedding(torch.LongTensor(word))
print(embed)
print(embed.size())
