import torchvision
from torch.utils.tensorboard import SummaryWriter

data_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor()
 ])

train_set = torchvision.datasets.CIFAR10(root='./dataset/CIFAR10', transform=data_transform, train=True, download=False)
test_set = torchvision.datasets.CIFAR10(root='./dataset/CIFAR10', transform=data_transform, train=False, download=False)

# print(test_set[0])
#
# print(test_set.classes)
#
# img, target = test_set[0]
# print(img)
# print(target)
# print(test_set.classes[target])     # 找对应索引的标签
# img.show()

print(test_set[0])

writer = SummaryWriter('E:\\code\\python_code\\Pytorch_gogo\\cs231n-2019-assignments-master\\Pytorch_learning_小土堆\\logs')

for i in range(10):
    img, target = test_set[i]
    writer.add_image('test_set', img, i)

writer.close()
