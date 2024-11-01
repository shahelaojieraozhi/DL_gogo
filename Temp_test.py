# -*- coding: utf-8 -*-
"""
@Project ：DL_gogo 
@Time    : 2023/10/27 14:45
@Author  : Rao Zhi
@File    : Temp_test.py
@email   : raozhi@mails.cust.edu.cn
@IDE     ：PyCharm 

"""
import torch

src = torch.randint(0, 10, (4, 66, 4096), dtype=torch.int64)
sign_tokens = torch.randint(0, 10, (4, 32), dtype=torch.int64)
# sign_tokens_for_loss = torch.cat(
#     (torch.zeros((src.shape[0], src.shape[1]), dtype=torch.int64), sign_tokens),
#     dim=-1)
tgt_input = torch.randint(0, 10, (4, 39), dtype=torch.int64).cuda()
masks = torch.cat((torch.ones((src.shape[0], src.shape[1]), dtype=torch.int64).cuda(), tgt_input), dim=-1)
print(masks.shape)
