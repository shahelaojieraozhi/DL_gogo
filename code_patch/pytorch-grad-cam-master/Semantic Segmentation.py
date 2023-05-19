"""
python3
-- coding: utf-8 --
-------------------------------
@Author : RAO ZHI
@Email : raozhi@mails.cust.edu.cn
-------------------------------
@File : Semantic Segmentation.py
@Software : PyCharm
@Time : 2023/4/26 19:05
-------------------------------
"""

import warnings

from matplotlib import pyplot as plt

warnings.filterwarnings('ignore')
warnings.simplefilter('ignore')
from torchvision.models.segmentation import deeplabv3_resnet50
import torch
import torch.functional as F
import numpy as np
import requests
import torchvision
from PIL import Image
from pytorch_grad_cam.utils.image import show_cam_on_image, preprocess_image

# image_url = "https://farm1.staticflickr.com/6/9606553_ccc7518589_z.jpg"
# image = np.array(Image.open(requests.get(image_url, stream=True).raw))

image_path = "9606553_ccc7518589_z.jpg"
image = np.array(Image.open(image_path))

rgb_img = np.float32(image) / 255
input_tensor = preprocess_image(rgb_img,
                                mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
# Taken from the torchvision tutorial
# https://pytorch.org/vision/stable/auto_examples/plot_visualization_utils.html
model = deeplabv3_resnet50(pretrained=True, progress=False)
model = model.eval()

if torch.cuda.is_available():
    model = model.cuda()
    input_tensor = input_tensor.cuda()

output = model(input_tensor)
print(type(output), output.keys())

"""
This package assumes the model will output a tensor. Here, instead, it's returning a dictionarty with the "out" and "aux" keys, 
where the actual result we care about is in "out". This is a common issue with custom networks, 
sometimes the model outputs a tuple, for example, and maybe you care only about one of it's outputs.

To solve this we're going to wrap the model first.
"""


class SegmentationModelOutputWrapper(torch.nn.Module):
    def __init__(self, model):
        super(SegmentationModelOutputWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        return self.model(x)["out"]


model = SegmentationModelOutputWrapper(model)
output = model(input_tensor)
print(output)
# output_matrix = output.view(375, 500).detach().cpu().numpy()
output_set = output[:, 0, :, :]
output_set_matrix = output_set.view(375, 500).detach().cpu().numpy()

normalized_masks = torch.nn.functional.softmax(output, dim=1).cpu()
# normalized_masks = normalized_masks[:, 0, :, :]
# normalized_masks_matrix = normalized_masks.view(375, 500).detach().cpu().numpy()


sem_classes = [
    '__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus',
    'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike',
    'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
]
sem_class_to_idx = {cls: idx for (idx, cls) in enumerate(sem_classes)}

car_category = sem_class_to_idx["car"]
# xx = normalized_masks[0, :, :, :].argmax(axis=0)
car_mask = normalized_masks[0, :, :, :].argmax(axis=0).detach().cpu().numpy()
car_mask_uint8 = 255 * np.uint8(car_mask == car_category)
car_mask_float = np.float32(car_mask == car_category)

both_images = np.hstack((image, np.repeat(car_mask_uint8[:, :, None], 3, axis=-1)))
Image.fromarray(both_images)

plt.imshow(Image.fromarray(both_images))
plt.show()

from pytorch_grad_cam import GradCAM


class SemanticSegmentationTarget:
    def __init__(self, category, mask):
        self.category = category
        self.mask = torch.from_numpy(mask)
        if torch.cuda.is_available():
            self.mask = self.mask.cuda()

    def __call__(self, model_output):
        return (model_output[self.category, :, :] * self.mask).sum()


target_layers = [model.model.backbone.layer4]
targets = [SemanticSegmentationTarget(car_category, car_mask_float)]
with GradCAM(model=model,
             target_layers=target_layers,
             use_cuda=torch.cuda.is_available()) as cam:
    grayscale_cam = cam(input_tensor=input_tensor,
                        targets=targets)[0, :]
    cam_image = show_cam_on_image(rgb_img, grayscale_cam, use_rgb=True)

Image.fromarray(cam_image)
plt.imshow(Image.fromarray(cam_image))
plt.show()
