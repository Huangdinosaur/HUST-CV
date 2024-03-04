import functorch.dim
import torch
import torchvision.models as models
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import torch.nn as nn
from PIL import Image
import numpy as np
import torch.nn.functional as F
import cv2
from torchvision.utils import make_grid
def show_feature_map_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    # print(mask.shape) (224, 224)
    # 转化为彩色图片
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    # print(heatmap.shape) (224, 224, 3)
    if use_rgb:
        # BGR转化为RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    # print(cam.shape)  (224, 224, 3)
    return np.uint8(255 * cam)




model = torch.load('./experiment4_data/torch_alex.pth')




# 加载并预处理输入图像
image_path = "./experiment4_data/data4/both.jpg"
image = Image.open(image_path).convert('RGB')
# 转化为np数组，每个元素取值为0~255，即uint8
img = np.array(image, dtype=np.uint8)
# print(img)


preprocess = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])
input_tensor = preprocess(image).unsqueeze(0)
# print(input_tensor)
# 将输入图像传递给模型以获取特征图
model.eval()
with torch.no_grad():
    features = model.features(input_tensor)

print(F.softmax(model(input_tensor), dim=1))
# 可视化每个通道上的特征图
# print(features.size(1))
# 可视化每个通道上的特征图
images = []
for channel_idx in range(features.size(1)):
    feature_map = features[0, channel_idx, :, :].numpy()
    # 将特征图的值归一化到0~1
    # print(feature_map.shape) (6, 6)
    # 特征值取绝对值？
    # feature_map = np.abs(feature_map)
    # 抛弃负值!
    feature_map = np.where(feature_map >= 0, feature_map, 0)
    base = np.max(feature_map)-np.min(feature_map)
    if base != 0:
        feature_map = (feature_map - np.min(feature_map))/base
    feature_map = cv2.resize(feature_map, (224, 224))
    visualization = show_feature_map_on_image(img.astype(dtype=np.float32) / 255.,
                                      feature_map,
                                      use_rgb=True)
    images.append(visualization)
    # plt.imshow(visualization)
    # plt.axis('off')
    # plt.show()
#将获取得到的所有图片即images List转化为np数组，再更改维度  图片数量：宽：高：通道→图片数量：通道：宽：高
tensor_images = torch.from_numpy(np.array(images).transpose((0, 3, 1, 2)))
# print(tensor_images.shape)
# 16张一行
grid_image = make_grid(tensor_images, nrow=16)
# print(grid_image.shape) torch.Size([3, 3618, 3618])
# 将元素的维度转化回去
grid_image = grid_image.permute(1, 2, 0).numpy()
# 展示
plt.imshow(grid_image)
plt.axis('off')
plt.show()