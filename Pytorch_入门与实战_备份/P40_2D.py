import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

'使用一张图像来展示经过卷积后的图像效果'

# # 读取图像－转化为灰度图片－转化为numpy数组
myim = Image.open("./11.jpg")
myimgray = np.array(myim.convert("L"), dtype=np.float32)
# # 可视化图片
plt.figure(figsize=(6, 6))
plt.imshow(myimgray, cmap=plt.cm.gray)
plt.axis("off")
plt.show()

h, w = myimgray.shape
print('shape( h * w )：', h, w)

# # 对灰度图像进行卷积提取图像轮廓
# # 将数组转化为张量
imh, imw = myimgray.shape
myimgray_t = torch.from_numpy(myimgray.reshape((1, 1, imh, imw)))
print('myimgray_t.shape：', myimgray_t.shape)
# # 因为卷积时需要操作一张图像，所以将图像转化为4维表示［batch，channel,h,w］

# # 定义边缘检测卷积核,并纬度处理为1*1*5*5
kersize = 5
ker = torch.ones(kersize, kersize, dtype=torch.float32) * -1
ker[2, 2] = 24
ker = ker.reshape((1, 1, kersize, kersize))
# # 进行卷积操作
conv2d = nn.Conv2d(1, 1, (kersize, kersize), bias=False)
# # 设置卷积时使用的核
conv2d.weight.data = ker
# # 对灰度图像进行卷积操作
imconv2dout = conv2d(myimgray_t)
# # 对卷积后的输出进行维度压缩
imconv2dout_im = imconv2dout.data.squeeze()
print("卷积后尺寸:", imconv2dout_im.shape)  # torch.Size([366, 274])
# # 查看使用的卷积核
print('使用的卷积核:', conv2d.weight.data)
# 使用的卷积核: tensor([[[[-1., -1., -1., -1., -1.],
#           [-1., -1., -1., -1., -1.],
#           [-1., -1., 24., -1., -1.],
#           [-1., -1., -1., -1., -1.],
#           [-1., -1., -1., -1., -1.]]]])


# # 可视化卷积后的图像
plt.figure(figsize=(6, 6))
plt.imshow(imconv2dout_im, cmap=plt.cm.gray)
plt.axis("off")
plt.show()

# # 定义边缘检测卷积核,并纬度处理为1*1*5*5
kersize = 5
ker = torch.ones(kersize, kersize, dtype=torch.float32) * -1
ker[2, 2] = 24
ker = ker.reshape((1, 1, kersize, kersize))
# # 进行卷积操作
conv2d = nn.Conv2d(1, 2, (kersize, kersize), bias=False)
# # 设置卷积时使用的核,第一个核使用边缘检测核
conv2d.weight.data[0] = ker
# # 对灰度图像进行卷积操作
imconv2dout = conv2d(myimgray_t)
# # 对卷积后的输出进行维度压缩
imconv2dout_im = imconv2dout.data.squeeze()
print("卷积后尺寸:", imconv2dout_im.shape)   # torch.Size([2, 366, 274])

# # 可视化卷积后的图像
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.imshow(imconv2dout_im[0], cmap=plt.cm.gray)
plt.axis("off")
plt.subplot(1, 2, 2)
plt.imshow(imconv2dout_im[1], cmap=plt.cm.gray)
plt.axis("off")
plt.show()

