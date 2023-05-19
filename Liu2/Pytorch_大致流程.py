import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch import nn

data = pd.read_csv(r'./dataset/Pytoch_data/Income1.csv')
print(data.info)

# pytorch 默认处理tensor类型数据

print(data.Education.values)
# 但是这样输出 返回的是array类型


# data.Education.values 相当于把pandas的series转换成了numpy的series
data1 = data.Education.values.reshape(-1, 1).astype(np.float32)
X = torch.from_numpy(data1)

data2 = data.Income.values.reshape(-1, 1).astype(np.float32)
Y = torch.from_numpy(data2)

model = nn.Linear(1, 1)
# 相当于out =  w*Input + b
# Linear会随机产生一个权重w和一个bias  等价于model(input)

loss_fn = nn.MSELoss()  # 因为它是个类所以加个括号
# 优化算法
opt = torch.optim.SGD(model.parameters(), lr=0.0001)

# epoch 代表一次训练全部数据
for epoch in range(5000):
    for x, y in zip(X, Y):
        y_pred = model(x)  # 使用模型预测
        loss = loss_fn(y, y_pred)  # 根据预测结果计算损失
        opt.zero_grad()  # 把变量梯度清0
        loss.backward()  # 求解梯度
        opt.step()  # 优化模型参数

print(model.weight, model.bias)

plt.scatter(data.Education, data.Income)
# X.numpy()能把tensor转换成numpy类型，不然画不了图
plt.plot(X.numpy(), model(X).data.numpy(), c='r')
# model(X)里面还有梯度信息，我们只要提取出data

plt.show()
