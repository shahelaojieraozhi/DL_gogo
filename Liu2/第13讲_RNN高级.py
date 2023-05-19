import torch
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision import datasets
import torch.nn.functional as F
import torch.optim as optim
import gzip
import csv
import time
import math

# ###参数设置
HIDDEN_SIZE = 100
BATCH_SIZE = 256
N_LAYER = 2  # GRU用了两层
N_EPOCHS = 30
N_CHARS = 128  # 输入字典长度
USE_GPU = True


class NameDataset(Dataset):
    def __init__(self, is_train_set=True):
        # train 和 test 分开了，所以可以这样选择
        filename = './dataset/names_train.csv.gz' if is_train_set else './dataset/names_test.csv.gz'
        with gzip.open(filename, 'rt') as f:
            reader = csv.reader(f)
            rows = list(reader)
        # 分别获得list形式的name和country
        self.names = [row[0] for row in rows]  # rows = [[name1, country1],[name2, country2]......]
        self.len = len(self.names)
        self.countries = [row[1] for row in rows]

        # # 创建城市字典表，set是去掉重复的城市+排序(这个排序应该默认为)
        self.country_list = list(sorted(set(self.countries)))
        self.country_dict = self.getCountryDict()  # 做一个词典表
        self.country_num = len(self.country_list)

    def __getitem__(self, index):
        return self.names[index], self.country_dict[self.countries[index]]
        # Country相当于 key, Index 相当于 value

    def __len__(self):
        return self.len

    def getCountryDict(self):
        country_dict = dict()
        for idx, country_name in enumerate(self.country_list, 0):
            country_dict[country_name] = idx
        return country_dict

    # country_list = [Arabic, Chinese, Czech, Dutch.....]
    # country_dict = {'Arabic': 0, 'Chinese': 1, 'Czech': 2, 'Dutch': 3......}

    def idx2country(self, index):
        # 通过索引找到城市名,应该用于是后期返回值是某个国家(城市)
        return self.country_list[index]

    def getCountriesNum(self):
        # 有多少个国家？
        return self.country_num


# 数据准备工作
trainset = NameDataset(is_train_set=True)
trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True)
testset = NameDataset(is_train_set=False)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False)

N_COUNTRY = trainset.getCountriesNum()  # 决定模型将来输出的大小


# ####模型设计
class RNNClassifier(torch.nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, bidirectional=True):
        super(RNNClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.n_directions = 2 if bidirectional else 1
        # 双向还是单向, bidirectional=True, 所以self.n_directions=2
        self.embedding = torch.nn.Embedding(input_size, hidden_size)  # 嵌入层
        self.gru = torch.nn.GRU(hidden_size, hidden_size, n_layers, bidirectional=bidirectional)
        '''
        torch.nn.GRU的输入和输出都是hidden_size
        n_layers 决定用了几层GRU
        bidirectional的值决定是单向向还是双向GRU
        如果是双向GRU的hidden返回[正向h, 反向h]
        '''
        # bidirectional决定是单向向还是双向GRU
        self.fc = torch.nn.Linear(hidden_size * self.n_directions, output_size)

    def __init_hidden(self, batch_size):
        hidden = torch.zeros(self.n_layers * self.n_directions, batch_size, self.hidden_size)
        return create_tensor(hidden)

    def forward(self, input, seq_lengths):
        # input shape: B * S -> S * B
        input = input.t()  # input.t()相当于input.transpose()
        batch_size = input.size(1)

        hidden = self.__init_hidden(batch_size)
        embedding = self.embedding(input)

        # pack them up
        gru_input = torch.nn.utils.rnn.pack_padded_sequence(embedding, seq_lengths)
        '''
        将embedding打包成sequence(GRU、RNN、LSTM都可以接受这个操作)
        加速运算，
        先把embedding排序把为0的全部舍弃掉，但记录不为0的长度
        '''

        output, hidden = self.gru(gru_input, hidden)
        # output返回值：正向和反向累contact的h
        # hidden返回值：如果是双向GRU返回[正向h, 反向h]

        '''如果是双层contact这两个hidden'''
        if self.n_directions == 2:
            hidden_cat = torch.cat([hidden[-1], hidden[-2]], dim=1)
        else:
            hidden_cat = hidden[-1]

        fc_output = self.fc(hidden_cat)
        return fc_output


# 功能函数构建
def make_tensors(names, countries):
    # 由名字转换成Tensor
    sequences_and_lengths = [name2list(name) for name in names]
    name_sequences = [sl[0] for sl in sequences_and_lengths]  # 先拿出名字的ASCII码列表
    seq_lengths = torch.LongTensor([sl[1] for sl in sequences_and_lengths])
    # 再拿出每个名字的ASCII码列表的的长度
    countries = countries.long()

    # make tensor of name, BatchSize x SeqLen
    seq_tensor = torch.zeros(len(name_sequences), seq_lengths.max()).long()
    # 创建全零Tensor(), shape为(名字的ASCII码列表的个数——几个人, 名字长度最长的)
    for idx, (seq, seq_len) in enumerate(zip(name_sequences, seq_lengths), 0):
        '''
        padding要把[77, 97, 108, 101, 97, 110] 转成 [77, 97, 108, 101, 97, 110, 0, 0, 0]
        seq_tensor 现在是[0, 0, 0, 0, 0, 0, 0, 0, 0],又知道seq_lengths(赋为了seq_len)=6
        seq_tensor[0, :6] = torch.LongTensor([77, 97, 108, 101, 97, 110])
        '''
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # sort by length to use pack_padded_sequence
    seq_lengths, perm_idx = seq_lengths.sort(dim=0, descending=True)
    # pytorch 的sort返回两个值：排完序的seq和对应的索引
    # 将seq_lengths按序列长度重新降序排序，返回排序结果和排序序列。
    seq_tensor = seq_tensor[perm_idx]  # 按新序列把ASCII表重新排序
    countries = countries[perm_idx]  # 按新序列把国家列表重新排序

    # 返回排序后的 ASCII列表         名字长度降序列表        国家名列表
    return create_tensor(seq_tensor), seq_lengths, create_tensor(countries)


'''
直接 create_tensor(seq_lengths) 会报错
# RuntimeError: 'lengths' argument should be a 1D CPU int64 tensor, but got 1D cuda:0 Long tensor

# "\"可以换行
'''


def name2list(name):
    '''
    输入'名字'返回名字的ASCII和长度
    input：name2list('Raozhi')
    return：([82, 97, 111, 122, 104, 105], 6)
    '''
    arr = [ord(c) for c in name]
    return arr, len(arr)


def time_since(since):
    s = time.time() - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def create_tensor(tensor):
    if USE_GPU:
        device = torch.device("cuda:0")
        tensor = tensor.to(device)
    return tensor


# 训练
def trainModel():
    total_loss = 0
    for i, (names, countries) in enumerate(trainloader, 1):
        inputs, seq_lengths, target = make_tensors(names, countries)
        output = classifier(inputs, seq_lengths)
        loss = criterion(output, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        if i % 10 == 0:
            print(f'[{time_since(start)}] Epoch{epoch}', end='')
            print(f'[{i * len(inputs)}/{len(trainset)}]', end='')
            print(f'loss={total_loss / (i * len(inputs))}')

    return total_loss


# 测试
def testModel():
    correct = 0
    total = len(testset)
    print("evaluating trained model ...")
    with torch.no_grad():
        for i, (names, countries) in enumerate(testloader, 1):
            inputs, seq_lengths, target = make_tensors(names, countries)
            output = classifier(inputs, seq_lengths)
            pred = output.max(dim=1, keepdim=True)[1]
            correct += pred.eq(target.view_as(pred)).sum().item()

        percent = '%.2f' % (100 * correct / total)
        print(f'Test set: Accuracy {correct}/{total} {percent}%')
    return correct / total


# 函数入口
if __name__ == '__main__':
    classifier = RNNClassifier(N_CHARS, HIDDEN_SIZE, N_COUNTRY, N_LAYER)
    if USE_GPU:
        device = torch.device("cuda:0")
        classifier.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(classifier.parameters(), lr=0.001)

    start = time.time()
    print("Training for %d epochs..." % N_EPOCHS)
    acc_list = []
    for epoch in range(1, N_EPOCHS + 1):
        trainModel()
        acc = testModel()
        acc_list.append(acc)

    epoch = np.arange(1, len(acc_list) + 1, 1)
    acc_list = np.array(acc_list)
    plt.plot(epoch, acc_list)
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid()
    plt.show()
