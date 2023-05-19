import torchvision
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
])

test_set = torchvision.datasets.CIFAR10(root='./dataset/CIFAR10', transform=data_transform, train=False, download=False)

test_loader = DataLoader(dataset=test_set, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

# 测试集第一张图片及target
img, target = test_set[0]
print(img.shape)
print(target)

writer = SummaryWriter('E:\\code\\python_code\\Pytorch_gogo\\cs231n-2019-assignments-master\\Pytorch_learning_小土堆\\logs')

# 取 Dataloader 里面的数
for epoch in range(2):
    step = 0
    for data in test_loader:
        imgs, targets = data
        # print(imgs.shape)
        # print(targets)
        writer.add_images("Epoch: {}".format(epoch), imgs, step)
        step += 1

writer.close()

