# -*- coding: utf-8 -*-
"""
@Project ：DL_gogo 
@Time    : 2023/10/19 15:07
@Author  : Rao Zhi
@File    : train.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils import *
from Network import *
import matplotlib.pyplot as plt

# hyperparams
enc_seq_len = 6
dec_seq_len = 2
output_sequence_length = 1

dim_val = 10
dim_attn = 5
n_heads = 3
n_decoder_layers = 3
n_encoder_layers = 3

lr = 0.002
epochs = 20
batch_size = 15

# init network and optimizer
t = Transformer(dim_val, dim_attn, 1, dec_seq_len, output_sequence_length, n_decoder_layers, n_encoder_layers, n_heads)
optimizer = torch.optim.Adam(t.parameters(), lr=lr)

# keep track of loss for graph
losses = []

for e in range(epochs):
    out = []

    for b in range(-10 - enc_seq_len, 10 - enc_seq_len):
        optimizer.zero_grad()
        X, Y = get_data(batch_size, enc_seq_len, output_sequence_length)

        # Forward pass and calculate loss
        net_out = t(X)
        loss = torch.mean((net_out - Y) ** 2)

        # backwards pass
        loss.backward()
        optimizer.step()

        # Track losses and draw rgaph
        out.append([net_out.cpu().detach().numpy(), Y])
        losses.append(loss.item())

plt.plot(losses)
plt.title("Mean Squared Error")
plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111)
# plt.ion()
#
# fig.show()
# fig.canvas.draw()
#
# o = []
# x = [torch.sigmoid(torch.arange(-10, -1).float()).unsqueeze(-1).numpy().tolist()]
#
# # Draw graph comparing to sigmoid
# for i in range(-10, 10, output_sequence_length):
#     o.append([torch.sigmoid(torch.tensor(i).float())])
#     q = torch.tensor(x).float()
#
#     if (output_sequence_length == 1):
#         x[0].append([t(q).detach().squeeze().numpy()])
#     else:
#         for a in t(q).detach().squeeze().numpy():
#             x[0].append([a])
#
# ax.clear()
# ax.plot(x[0], label='Network output')
# ax.plot(o, label='Sigmoid function')
# ax.set_title("")
# ax.legend(loc='upper left', frameon=False)
