# -*- coding: utf-8 -*-
"""
@Project ：DL_gogo 
@Time    : 2023/7/20 20:16
@Author  : Rao Zhi
@File    : Linear_torch.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import time

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torch import nn
from torch.utils import data
import seaborn as sns
from model import Mlp, TF
from dataset import MyDataSet
from torch.utils.tensorboard import SummaryWriter

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# ### Training


# ##### Define loss function and model
# ##### We can experiement with the model. For example, the simplest can be a single nn.Linear model

k, num_epochs, lr, weight_decay, batch_size = 5, 100, 0.002, 2, 256

mes_loss = nn.MSELoss()


def log_rmse(net, features, labels):
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(mes_loss(torch.log(clipped_preds), torch.log(labels)))
    return rmse.item()


model = TF(in_features=330, drop=0.).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

# ##### Load dataset
mydata = MyDataSet()
train_loader = data.DataLoader(mydata, batch_size=batch_size, shuffle=True)
in_features = len(train_loader)


def train_epoch():
    train_ls, it_count = 0, 0
    for batch_idx, train_data in enumerate(train_loader, 0):
        features, labels = train_data
        optimizer.zero_grad()
        outputs = model(features)
        loss = mes_loss(outputs, labels)

        loss.backward()
        optimizer.step()
        train_ls += loss.item()
        it_count += 1

    return train_ls / it_count


if __name__ == '__main__':
    count = 0
    writer = SummaryWriter('./logs')
    for epoch in range(1000):
        count += 1
        train_loss = train_epoch()
        writer.add_scalar("train_loss", train_loss, count)  # add_scalar 添加标量
        print("%d, loss: %.3e" % (count, train_loss))

    writer.close()

# def test():
#     test_ls, it_count = 0, 0
#     for _, train_data in enumerate(train_loader, 0):
#         features, labels = train_data
#         optimizer.zero_grad()
#         outputs = model(features)
#         loss = mes_loss(outputs, labels)
#
#         test_ls += loss.item()
#         it_count += 1
#
#     return test_ls / it_count


# train_ls_all = []
# train_ls_all.append(train_ls)
# # plt.plot(np.arange(1,101,1),train_ls_all[0])
# # plt.xlabel('epoch'), plt.ylabel('rmse')
# print(f'train log rmse {float(train_ls[-1]):f}')
# preds = net(test_features).cpu().detach().numpy()
# test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
# submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
# submission.to_csv('submission.csv', index=False)
