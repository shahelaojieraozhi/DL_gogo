import torch
#
# x = torch.rand(2, 3)
# print(x)
# x = torch.randn(3, 4)
# print(x)
# x = torch.ones(2, 3, 4)  # 可以理解为2个3*4的矩阵
# print(x)
#
# y = torch.ones((2, 3, 4), dtype=torch.int32)  # 可以理解为2个3*4的矩阵
# print(y)
#
# print(x.size())
# print(x.shape)
#
# x = torch.tensor([6, 2], dtype=torch.float32)
# x.type()
# # 数据类型的转换
# x.type(torch.int64)


import numpy as np
a = np.random.randn(2, 3)
print(a)

x1 = torch.from_numpy(a)
print(x1)

x2 = x1.numpy()
print(x2)

x2 = torch.rand(2, 3)
print(x2)

# tensor类型，对应元素相加
print(x1 + x2)

# 广播
print(x2 + 3)

