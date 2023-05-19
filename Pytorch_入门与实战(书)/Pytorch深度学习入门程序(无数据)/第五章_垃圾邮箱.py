# # 导入本章所需要的模块
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.manifold import TSNE

import torch
import torch.nn as nn
from torch.optim import SGD, Adam
import torch.utils.data as Data

import matplotlib.pyplot as plt
import seaborn as sns
import hiddenlayer as hl

# # 读取数据显示数据的前几行
spam = pd.read_csv("E:\\data_set\\spambase\\spambase.csv")
spam.head()

# # 计算垃圾邮件和非垃圾邮件的数量
# label_info = pd.value_counts(spam.label)
# print(label_info)

# # 将数据随机切分为训练集和测试集
X = spam.iloc[:, 0:57].values
y = spam.label.values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=123)

# # 对数据的前57列特征进行数据标准化处理
scales = MinMaxScaler(feature_range=(0, 1))
X_train_s = scales.fit_transform(X_train)
X_test_s = scales.transform(X_test)

# # 使用训练数据集对数据特征进行可视化
# # 使用密度曲线对比不同类别在每个特征上的数据分布情况
# feature_name 记录所有特征的名字
feature_name = spam.columns.values[:-1]

"""核密度估计并可视化----大图呈现"""
# # 详解seaborn可视化中的kdeplot、rugplot、distplot与jointplot :https://cloud.tencent.com/developer/article/1971963
plt.figure(figsize=(20, 14))
for i in range(len(feature_name)):
    plt.subplot(7, 9, i + 1)
    # X_train_s.shape = (3450, 57)
    # X_train_s[y_train == 0, i] 抽取X_train_s中第i个特征且标签为0的数据为x
    sns.kdeplot(X_train_s[y_train == 0, i], bw_method=0.05)
    # X_train_s[y_train == 1, i] 抽取X_train_s中第i个特征且标签为1的数据为x
    sns.kdeplot(X_train_s[y_train == 1, i], bw_method=0.05)
    plt.title(feature_name[i])
plt.subplots_adjust(hspace=1)
# hspace：子图间高度内边距，距离单位为子图平均高度的比例(小数)。可选参数。浮点数。默认值为0.2。
plt.show()

"""核密度估计并可视化----单个查看"""
# plt.figure(figsize=(8, 6))
# for i in range(len(feature_name)):
#     sns.kdeplot(X_train_s[y_train == 0, i], bw_method=0.05, color='r')
#     sns.kdeplot(X_train_s[y_train == 1, i], bw_method=0.05, color='b')
#     plt.title(feature_name[i])
#     plt.show()

""" 箱线 -整体实现"""
# 箱形图（Box-plot）：
# 又称为盒须图、盒式图或箱线图，是一种用作显示一组数据分散情况资料的统计图。它能显示出一组数据的最大值、最小值、中位数及上下四分位数

# # 使用训练数据集对数据特征进行可视化
# # 使用箱线对比不同类别在每个特征上的数据分布情况

# plt.figure(figsize=(20, 14))
# # plt.figure(figsize=(30, 20))
# # figsize=(30, 20) 不一样在jupyter上呈现的效果居然不一样
# for i in range(len(feature_name)):
#     plt.subplot(7, 9, i + 1)
#     sns.boxplot(x=y_train, y=X_train_s[:, i])
#     plt.title(feature_name[i])
# plt.subplots_adjust(hspace=0.4)
# plt.show()

""" 箱线---单独显示 """
# plt.figure(figsize=(8, 6))
# for i in range(len(feature_name)):
#     sns.boxplot(x=y_train, y=X_train_s[:, i])
#     plt.title(feature_name[i])
#     plt.show()

"""将数据转化为张量"""
X_train_t = torch.from_numpy(X_train_s.astype(np.float32))
y_train_t = torch.from_numpy(y_train.astype(np.int64))
X_test_t = torch.from_numpy(X_test_s.astype(np.float32))
y_test_t = torch.from_numpy(y_test.astype(np.int64))
# # 将训练集转化为张量后,使用TensorDataset将X和Y整理到一起
train_data = Data.TensorDataset(X_train_t, y_train_t)
# # 定义一个数据加载器，将训练数据集进行批量处理
train_loader = Data.DataLoader(
    dataset=train_data,  # 使用的数据集
    batch_size=64,  # 批处理样本大小
    shuffle=True,  # 每次迭代前打乱数据
    num_workers=1,  # 使用两个进程
)


# # 全连接网络
class MLPclassifica(nn.Module):
    def __init__(self):
        super(MLPclassifica, self).__init__()
        # # 定义第一个隐藏层
        self.hidden1 = nn.Sequential(
            nn.Linear(
                in_features=57,  # # 第一个隐藏层的输入，数据的特征数
                out_features=30,  # # 第一个隐藏层的输出，神经元的数量
                bias=True,  # # 默认会有偏置
            ),
            nn.ReLU()
        )
        # # 定义第二个隐藏层
        self.hidden2 = nn.Sequential(
            nn.Linear(30, 10),
            nn.ReLU()
        )
        # # 分类层
        self.classifica = nn.Sequential(
            nn.Linear(10, 2),
            nn.Sigmoid()
        )

    # # 定义网络的向前传播路径
    def forward(self, x):
        fc1 = self.hidden1(x)
        fc2 = self.hidden2(fc1)
        output = self.classifica(fc2)
        # # 输出为两个隐藏层和输出层的输出
        return fc1, fc2, output


# # 输出我们的网络结构
mlpc = MLPclassifica()
print(mlpc)

# 定义优化器
optimizer = torch.optim.Adam(mlpc.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()  # 二分类损失函数
# 记录训练过程的指标
history1 = hl.History()
# 使用Canvas进行可视化
canvas1 = hl.Canvas()
print_step = 25

# # 对模型进行迭代训练,对所有的数据训练EPOCH轮
for epoch in range(15):
    # # 对训练数据的迭代器进行迭代计算
    for step, (b_x, b_y) in enumerate(train_loader):
        # # 计算每个batch的
        _, _, output = mlpc(b_x)  # MLP在训练batch上的输出
        train_loss = loss_func(output, b_y)  # 二分类交叉熵损失函数
        optimizer.zero_grad()  # 每个迭代步的梯度初始化为0
        train_loss.backward()  # 损失的后向传播，计算梯度
        optimizer.step()  # 使用梯度进行优化
        niter = epoch * len(train_loader) + step + 1

        # # 计算每经过print_step次迭代后的输出
        if niter % print_step == 0:
            _, _, output = mlpc(X_test_t)
            _, pre_lab = torch.max(output, 1)
            test_accuracy = accuracy_score(y_test_t, pre_lab)
            # 为history添加epoch，损失和精度
            history1.log(niter, train_loss=train_loss,
                         test_accuracy=test_accuracy)
            # 使用两个图像可视化损失函数和精度
            with canvas1:
                canvas1.draw_plot(history1["train_loss"])
                canvas1.draw_plot(history1["test_accuracy"])

""" 损失使用平均值的输出 """
# 定义优化器
optimizer = torch.optim.Adam(mlpc.parameters(), lr=0.01)
loss_func = nn.CrossEntropyLoss()  # 二分类损失函数
# 记录训练过程的指标
history1 = hl.History()
# 使用Canvas进行可视化
canvas1 = hl.Canvas()
print_step = 25
train_loss_all = 0
# # 对模型进行迭代训练,对所有的数据训练EPOCH轮
for epoch in range(15):
    # # 对训练数据的迭代器进行迭代计算
    for step, (b_x, b_y) in enumerate(train_loader):
        ## 计算每个batch的
        _, _, output = mlpc(b_x)  # MLP在训练batch上的输出
        train_loss = loss_func(output, b_y)  # 二分类交叉熵损失函数
        optimizer.zero_grad()  # 每个迭代步的梯度初始化为0
        train_loss.backward()  # 损失的后向传播，计算梯度
        optimizer.step()  # 使用梯度进行优化
        niter = epoch * len(train_loader) + step + 1
        train_loss_all += train_loss

        # # 计算每经过print_step次迭代后的输出
        if niter % print_step == 0:
            _, _, output = mlpc(X_test_t)
            _, pre_lab = torch.max(output, 1)
            test_accuracy = accuracy_score(y_test_t, pre_lab)
            # 为history添加epoch，损失和精度
            history1.log(niter, train_loss=train_loss / niter,
                         test_accuracy=test_accuracy)
            # 使用两个图像可视化损失函数和精度
            with canvas1:
                canvas1.draw_plot(history1["train_loss"])
                canvas1.draw_plot(history1["test_accuracy"])


# # 计算最终模型在测试集上的精度
_, _, output = mlpc(X_test_t)
_, pre_lab = torch.max(output, 1)
test_accuracy = accuracy_score(y_test_t, pre_lab)
print("test_accuracy:", test_accuracy)
print(classification_report(y_test_t, pre_lab))
print(confusion_matrix(y_test_t, pre_lab))
