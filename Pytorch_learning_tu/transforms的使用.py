from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter

# tips: ctrl + P 可以查看要选的参数
# 设置大小写通用，setting-》
'''
python的用法 ——》 tensor数据类型

通过 transform.ToTensor 去看两个问题
1.transform 该如何使用(python)
2.为什么要使用tensor的类型
'''

img_path = './ant1.jpg'
img = Image.open(img_path)
# print(img)

writer = SummaryWriter("logs")
tensor_trans = transforms.ToTensor()  # 实例化 transforms.ToTensor()类
tensor_img = tensor_trans(img)
print(tensor_img)

# writer.add_image("tensor_img", tensor_img)
# writer.close()

# transforms.Normalize()
trans_norm = transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
img_norm = trans_norm(tensor_img)
print(img_norm[0][0][0])

# Resize()
print(img.size)
trans_resize = transforms.Resize((512, 512))

# img  PIL -> resize -> img_size  PIL
img_resize = trans_resize(img)
# img_resize PIL -> to_tensor -> img_resize tensor
img_resize = tensor_trans(img_resize)
print(img_resize)
# img_resize.show()

# Compose -resize -2
trans_resize_2 = transforms.Resize(512)
# PIL -> PIL -> tensor
trans_compose = transforms.Compose([trans_resize_2, tensor_trans])
img_resize_2 = trans_compose(img)

writer.add_image("Resize", img_resize_2, 1)
writer.close()

# 启动命令： tensorboard --logdir “绝对路径”


'''
总结：
关注输入和输出类型
多看官方文档(官方文档最清楚)
关注方法需要什么参数

不知道返回值的时候
print()
print(type())
debug

'''