from torch.utils.data import Dataset
import cv2
from PIL import Image
import os


class MyData(Dataset):

    def __init__(self, root_dir, label_dir):
        # 这个up主对self.xx 的理解是：相当于定义一个全局变量，普通的变量在函数之间不能相互共用

        self.root_dir = root_dir
        self.label_dir = label_dir
        self.path = os.path.join(self.root_dir, self.label_dir)
        # 获取所有图片的列表
        self.img_path = os.listdir(self.path)

    def __getitem__(self, item):
        img_name = self.img_path[item]
        img_item_path = os.path.join(self.root_dir, self.label_dir, img_name)
        img = Image.open(img_item_path)
        label = self.label_dir
        return img, label

    def __len__(self):
        return len(self.img_path)


# img_path = r"E:\code\python_code\Pytorch_gogo\cs231n-2019-assignments-master\Pytorch_learning_小土堆\ant1.jpg"
# img = Image.open(img_path)
# # img.show()

root_dir = r"E:\code\python_code\Pytorch_gogo\cs231n-2019-assignments-master\Pytorch_learning_小土堆\dataset/train"
ants_label_dir = "ants"
bees_label_dir = 'bees'
ants_dataset = MyData(root_dir, ants_label_dir)
bees_dataset = MyData(root_dir, bees_label_dir)

print(ants_dataset[1])  # 传出 img 和 label
# (<PIL.JpegImagePlugin.JpegImageFile image mode=RGB size=768x512 at 0x1E126466D08>, 'ants')

# img, label = ants_dataset[1]
#
# img.show()

train_dataset = ants_dataset + bees_dataset
print(len(train_dataset))   # 245 = 124 + 121
