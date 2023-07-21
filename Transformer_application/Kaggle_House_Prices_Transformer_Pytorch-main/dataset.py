# -*- coding: utf-8 -*-
"""
@Project ：DL_gogo 
@Time    : 2023/7/20 20:17
@Author  : Rao Zhi
@File    : dataset.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils import data
import seaborn as sns
from model import Mlp, TF
import torch.utils.data as Data

# ### Import Data
train_raw = pd.read_csv('train.csv')
test_raw = pd.read_csv('test.csv')

# ##### description
# print(train_raw.shape, test_raw.shape)
# print(train_raw.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]])

# ##### Concatenate the train and test data for standardisation
all_features_raw = pd.concat((train_raw.iloc[:, 1:-1], test_raw.iloc[:, 1:]))
all_features = all_features_raw.copy()

# ### Data Preprocessing
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features[numeric_features] = all_features[numeric_features].fillna(0)

# ##### Get Dummies

all_features = pd.get_dummies(all_features, dummy_na=True, dtype=float)
print("train+test features data [samples_num(train + test), features_nums]: ", all_features.shape)
# print(all_features)

# heatmap
# corrmat = all_features.corr()
# plt.subplots(figsize=(15, 15))
# sns.heatmap(corrmat, vmax=0.9, square=True)
# plt.savefig('./corr.png')
# plt.show()

# ###### Convert to tensors
device = 'cuda' if torch.cuda.is_available() else 'cpu'

n_train = train_raw.shape[0]
test_features = torch.tensor(all_features[n_train:].values, dtype=torch.float32).to(device)


class MyDataSet(Data.Dataset):
    def __init__(self):
        self.train_features = torch.tensor(all_features[:n_train].values, dtype=torch.float32).to(device)
        self.train_labels = torch.tensor(train_raw.SalePrice.values.reshape(-1, 1), dtype=torch.float32).to(device)

    def __getitem__(self, idx):
        return self.train_features[idx], self.train_labels[idx]

    def __len__(self):
        return self.train_features.shape[0]  # two sentences

# loader = Data.DataLoader(MyDataSet(enc_inputs, dec_inputs, dec_outputs), 2, True)
