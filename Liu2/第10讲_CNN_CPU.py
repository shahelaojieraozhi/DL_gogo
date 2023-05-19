import torch
from torchvision import transforms
from torchvision import datasets
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.optim as optim

# prepare dataset

batch_size = 64
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])

train_dataset = datasets.MNIST(root='./dataset/mnist/', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
test_dataset = datasets.MNIST(root='./dataset/mnist/', train=False, download=True, transform=transform)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=batch_size)


# design model using class


class Net(torch.nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = torch.nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = torch.nn.Conv2d(10, 20, kernel_size=5)
        self.pooling = torch.nn.MaxPool2d(2)
        self.fc = torch.nn.Linear(320, 10)

    def forward(self, x):
        # flatten data from (n,1,28,28) to (n, 784)
        batch_size = x.size(0)  # x是(n,1,28,28) n为样本个数，一个batch的样本个数
        x = F.relu(self.pooling(self.conv1(x)))
        x = F.relu(self.pooling(self.conv2(x)))
        x = x.view(batch_size, -1)  # -1 此处自动算出的是320
        x = self.fc(x)

        return x


model = Net()

# construct loss and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.5)


# training cycle forward, backward, update


def train(epoch):
    running_loss = 0.0
    for batch_idx, data in enumerate(train_loader, 0):
        inputs, target = data
        optimizer.zero_grad()

        outputs = model(inputs)
        loss = criterion(outputs, target)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if batch_idx % 300 == 299:
            print('[%d, %5d] loss: %.3f' % (epoch + 1, batch_idx + 1, running_loss / 300))
            running_loss = 0.0


def test():
    correct = 0
    total = 0
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            outputs = model(images)
            _, predicted = torch.max(outputs.data, dim=1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    print('accuracy on test set: %d %% ' % (100 * correct / total))


if __name__ == '__main__':
    import time
    for epoch in range(10):
        T1 = time.time()
        train(epoch)
        test()
        T2 = time.time()
        print('训练一个epoch的时间:%s毫秒' % ((T2 - T1) * 1000))  # 程序运行时间:0.0毫秒

'''知识点——Convolution_layer'''
'''N input channels and M Output channels'''
# import torch
#
# in_channels, out_channels = 5, 10
# width, height = 100, 100
# kernel_size = 3
# batch_size = 1
#
#
# input = torch.randn(batch_size,
#                     in_channels,
#                     width,
#                     height)
# # 生成了[1, 5, 100, 100]的张量
#
# conv_layer = torch.nn.Conv2d(in_channels,
#                              out_channels,
#                              kernel_size=kernel_size)
#
# output = conv_layer(input)
#
# print(input.shape)
# print(output.shape)
# print(conv_layer.weight.shape)

# '''padding'''
# import torch
#
# input = [3, 4, 6, 5, 7, 2, 4, 6, 8, 2, 1, 6, 7, 8, 4, 9, 7, 4, 6, 2, 3, 7, 5, 4, 1]
# input = torch.Tensor(input).view(1, 1, 5, 5)
# conv_layer = torch.nn.Conv2d(1, 1, kernel_size=3, padding=1, bias=False)
# kernel = torch.Tensor([1, 2, 3, 4, 5, 6, 7, 8, 9]).view(1, 1, 3, 3)
# # 为什么要 conv_layer.weight.data = kernel.data   应该是指定卷积核的样子
# conv_layer.weight.data = kernel.data
# output = conv_layer(input)
# print(output)


# # Debug
# A = np.array([[0, 0, 0], [0, 3, 4], [0, 2, 4]])
# print(A)
# B = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(3, 3)
# print(B)
# print(A * B)
# print(np.sum(A * B))
#
# # 91

'''Maxpooling'''

# import torch
#
# input = [3, 4, 6, 5, 7, 2, 4, 6, 8, 2, 1, 6, 7, 8, 4, 9, 7, 4, 6, 2, 3, 7, 5, 4, 1]
# input = torch.Tensor(input).view(1, 1, 5, 5)
# maxpooling_layer = torch.nn.MaxPool2d(kernel_size=2)
#
# output = maxpooling_layer(input)
# print(output)
