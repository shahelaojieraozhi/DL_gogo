# import torch
#
# batchsize = 1
# seq_len =3
# input_size = 4
# hidden_size = 2
#
# cell = torch.nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
# # (seq, batch, features)
# dataset = torch.randn(seq_len, batchsize, input_size)
# hidden = torch.zeros(batchsize, hidden_size)
#
# for idx, input in enumerate(dataset):
#     print('=' * 20, idx, '=' * 20)
#     print('Input size:', input.shape)
#
#     hidden = cell(input, hidden)
#
#     print('outputs size:', hidden.shape)
#     print(hidden)


# import torch
#
# batch_size = 1
# seq_len = 3
# input_size = 4
# hidden_size = 2
# num_layers = 1
# cell = torch.nn.RNN(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers)
# # 构造RNN时指明输入维度，隐层维度以及RNN的层数
# inputs = torch.randn(seq_len, batch_size, input_size)
# hidden = torch.zeros(num_layers, batch_size, hidden_size)
# out, hidden = cell(inputs, hidden)
# print('Output size:', out.shape)
# print('Output:', out)
# print('Hidden size:', hidden.shape)
# print('Hidden', hidden)


# # 使用RNNcell训练
# import torch.nn as nn
# import torch
#
# input_size = 4
# hidden_size = 4
# batch_size = 1
#
# idx2char = ['e', 'h', 'l', 'o']
# x_data = [1, 0, 2, 3, 3]  # hello中各个字符的下标
# y_data = [3, 1, 2, 3, 2]  # ohlol中各个字符的下标
#
# one_hot_lookup = [[1, 0, 0, 0],
#                   [0, 1, 0, 0],
#                   [0, 0, 1, 0],
#                   [0, 0, 0, 1]]
# x_one_hot = [one_hot_lookup[x] for x in x_data]  # (seqLen, inputSize)
#
# inputs = torch.Tensor(x_one_hot).view(-1, batch_size, input_size)
# labels = torch.LongTensor(y_data).view(-1, 1)  # torch.Tensor默认是torch.FloatTensor是32位浮点类型数据，torch.LongTensor是64位整型
# print(inputs.shape)
# print(labels.shape)
#
#
# class Model(nn.Module):
#     def __init__(self, input_size, hidden_size, batch_size):
#         super(Model, self).__init__()
#         self.batch_size = batch_size
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.rnncell = nn.RNNCell(input_size=self.input_size, hidden_size=self.hidden_size)
#
#     def forward(self, inputs, hidden):
#         hidden = self.rnncell(inputs, hidden)  # (batch_size, hidden_size)
#         return hidden
#
#     def init_hidden(self):
#         return torch.zeros(self.batch_size, self.hidden_size)
#
#
# net = Model(input_size, hidden_size, batch_size)
#
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.1)
#
# epochs = 15
#
# for epoch in range(epochs):
#     loss = 0
#     optimizer.zero_grad()
#     hidden = net.init_hidden()
#     print('Predicted string:', end='')
#     for input, label in zip(inputs, labels):
#         hidden = net(input, hidden)
#         # 注意交叉熵在计算loss的时候维度关系，这里的hidden是([1, 4]), label是 ([1])
#         loss += criterion(hidden, label)
#         _, idx = hidden.max(dim=1)
#         print(idx2char[idx.item()], end='')
#     loss.backward()
#     optimizer.step()
#     print(', Epoch [%d/15] loss=%.4f' % (epoch + 1, loss.item()))

# # 使用RNN
# import torch
#
# input_size = 4
# hidden_size = 4
# num_layers = 1
# batch_size = 1
# seq_len = 5
# # 准备数据
# idx2char = ['e', 'h', 'l', 'o']
# x_data = [1, 0, 2, 2, 3]  # hello
# y_data = [3, 1, 2, 3, 2]  # ohlol
#
# one_hot_lookup = [[1, 0, 0, 0],
#                   [0, 1, 0, 0],
#                   [0, 0, 1, 0],
#                   [0, 0, 0, 1]]  # 分别对应0,1,2,3项
# x_one_hot = [one_hot_lookup[x] for x in x_data]  # 组成序列张量
# print('x_one_hot:', x_one_hot)
#
# # 构造输入序列和标签
# inputs = torch.Tensor(x_one_hot).view(seq_len, batch_size, input_size)
# labels = torch.LongTensor(y_data)  # labels维度是: (seqLen * batch_size ，1)
#
#
# # design model
# class Model(torch.nn.Module):
#     def __init__(self, input_size, hidden_size, batch_size, num_layers=1):
#         super(Model, self).__init__()
#         self.num_layers = num_layers
#         self.batch_size = batch_size
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.rnn = torch.nn.RNN(input_size=self.input_size,
#                                 hidden_size=self.hidden_size,
#                                 num_layers=self.num_layers)
#
#     def forward(self, input):
#         hidden = torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
#         out, _ = self.rnn(input, hidden)
#         # 为了能和labels做交叉熵，需要reshape一下:(seqlen*batchsize, hidden_size),即二维向量，变成一个矩阵
#         return out.view(-1, self.hidden_size)
#
#
# net = Model(input_size, hidden_size, batch_size, num_layers)
#
# # loss and optimizer
# criterion = torch.nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(net.parameters(), lr=0.05)
#
# # train cycle
# for epoch in range(20):
#     optimizer.zero_grad()
#     # inputs维度是: (seqLen, batch_size, input_size) labels维度是: (seqLen * batch_size * 1)
#     # outputs维度是: (seqLen, batch_size, hidden_size)
#     outputs = net(inputs)
#     loss = criterion(outputs, labels)
#     loss.backward()
#     optimizer.step()
#
#     _, idx = outputs.max(dim=1)
#     idx = idx.data.numpy()
#     print('Predicted: ', ''.join([idx2char[x] for x in idx]), end='')
#     print(',Epoch [%d/20] loss=%.3f' % (epoch + 1, loss.item()))

# Embedding编码方式
import torch

input_size = 4
num_class = 4
hidden_size = 8
embedding_size = 10
batch_size = 1
num_layers = 2
seq_len = 5

idx2char_1 = ['e', 'h', 'l', 'o']
idx2char_2 = ['h', 'l', 'o']

x_data = [[1, 0, 2, 2, 3]]
y_data = [3, 1, 2, 2, 3]

# inputs 维度为（batchsize，seqLen）
inputs = torch.LongTensor(x_data)
# labels 维度为（batchsize*seqLen）
labels = torch.LongTensor(y_data)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        # 告诉input大小和 embedding大小 ，构成input_size * embedding_size 的矩阵
        self.emb = torch.nn.Embedding(input_size, embedding_size)

        self.rnn = torch.nn.RNN(input_size=embedding_size,
                                hidden_size=hidden_size,
                                num_layers=num_layers,
                                batch_first=True)
        # batch_first=True，input of RNN:(batchsize,seqlen,embeddingsize) output of RNN:(batchsize,seqlen,hiddensize)
        self.fc = torch.nn.Linear(hidden_size, num_class)  # 从hiddensize 到 类别数量的 变换

    def forward(self, x):
        hidden = torch.zeros(num_layers, x.size(0), hidden_size)
        x = self.emb(x)  # 进行embedding处理，把输入的长整型张量转变成嵌入层的稠密型张量
        x, _ = self.rnn(x, hidden)
        x = self.fc(x)
        return x.view(-1, num_class)  # 为了使用交叉熵，变成一个矩阵（batchsize * seqlen,numclass）


net = Model()

criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(net.parameters(), lr=0.05)

for epoch in range(15):
    optimizer.zero_grad()
    outputs = net(inputs)

    loss = criterion(outputs, labels)
    loss.backward()
    optimizer.step()

    _, idx = outputs.max(dim=1)
    idx = idx.data.numpy()
    print('Predicted string: ', ''.join([idx2char_1[x] for x in idx]), end='')
    print(", Epoch [%d/15] loss = %.3f" % (epoch + 1, loss.item()))
