import numpy as np
import pandas as pd
import cv2
from PIL import Image
import torch.nn as nn
import matplotlib.pyplot as plt
import torch
from torchvision import transforms
import torch.nn.functional as F
from pytorch_grad_cam import LayerCAM
from torchcam.utils import overlay_mask
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

# 加在模型
model = torch.load('./experiment4_data/torch_alex.pth')
model.eval()

data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor()])
img_path = "./experiment4_data/data4/both.jpg"
# 以RGB格式打开
img_ = Image.open(img_path).convert('RGB')
# [C, H, W]
img_tensor = data_transform(img_)
# expand batch dimension
# 转化为np数组，每个元素取值为0~255，即uint8
img = np.array(img_, dtype=np.uint8)
# [C, H, W] -> [N, C, H, W]
# 增加一个维度
input_tensor = torch.unsqueeze(img_tensor, dim=0)
pred_logits = model(input_tensor)
pred_softmax = F.softmax(pred_logits, dim=1)
print(pred_softmax)

targets = [ClassifierOutputTarget(1)]
target_layers = [model.features[-3]]
cam = LayerCAM(model=model, target_layers=target_layers)
cam_map = cam(input_tensor=input_tensor, targets=targets)[0]

# plt.imshow(cam_map)
# plt.show()



result = overlay_mask(img_, Image.fromarray(cam_map), alpha=0.5)
print(type(result))
# result.save('./output/B2.jpg')

plt.imshow(result)
plt.axis('off')  # 关闭坐标轴显示
plt.show()






