# write by myself
import cv2
import numpy as np
import torch
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn as nn
from torchvision import transforms

class ActivationsAndGradients:
    def __init__(self, model, target_layer):
        self.model = model
        self.gradients = 0
        self.activations = 0
        self.handles = []
        # 注册钩子函数，正向转播时，经过该layer将回调传入的函数
        self.handles.append(
            target_layer.register_forward_hook(self.save_activation)
        )
        self.handles.append(
            target_layer.register_full_backward_hook(
                self.save_gradient))
    def save_activation(self, module, input, output):
        activation = output
        # 分离一个新的出来，没有梯度
        self.activations = activation.detach()
    def save_gradient(self, module, grad_input, grad_output):
        # 获取反向梯度
        grad = grad_output[0]
        self.gradients = grad.detach()
    def __call__(self, x):
        self.gradients = 0
        self.activations = 0
        # 已经设置好众多钩子函数，训练时将自动触发，保存各层的A,A'，此时触发前向钩子
        return self.model(x)
class GradCAM:
    def __init__(self, model, target_layers):
        self.model = model.eval()
        self.target_layers = target_layers
        # 下面这个类，捕获特征层A，以及捕获反向梯度信息A'
        # 下面是该类里面的成员类，数据是时刻更新的
        self.activations_and_grads = ActivationsAndGradients(
            self.model, target_layers
        )
    def __call__(self, input_tensor, target_category=None):
        # 正向传播得到网络输出logits(未经过softmax)
        # 调用，收集A,A'
        output = self.activations_and_grads(input_tensor)
        # 变成一个矩阵，比如输入1，变为[1,1,1,1,1,...,1],具体长度取决于input_tensor
        if isinstance(target_category, int):
            target_category = [target_category] * input_tensor.size(0)
        # 清空梯度，防止累加
        self.model.zero_grad()
        # 收集指定类别的预测值，即y
        loss = self.get_predict(output, target_category)
        # 对各个预测值触发反向传播，开始求导并通过设置好的后向钩子函数保存到响应位置
        loss.backward(retain_graph=True)
        # 得到前向特征A,梯度A',然后开始计算cam
        return self.compute_cam_per_layer(input_tensor)
    @staticmethod
    def get_predict(output, target_category):
        loss = 0
        for i in range(len(target_category)):
            # i表示第几张图片，即收集指定类别的预测值
            # 即求yc
            loss = loss + output[i, target_category[i]]
        return loss
    @staticmethod
    def scale_cam_image(cam, target_size=None):
        result = []
        for img in cam:
            img = img - np.min(img)
            img = img / (1e-7 + np.max(img))
            # 恢复到原图尺寸
            if target_size is not None:
                img = cv2.resize(img, target_size)
            result.append(img)
        result = np.float32(result)
        return result
    def compute_cam_per_layer(self, input_tensor):
        # 提取收集到的A,A'
        layer_activations, layer_grads = [self.activations_and_grads.activations,
                                          self.activations_and_grads.gradients]
        # print(type(layer_grads))
        activations = layer_activations.numpy()
        grads = layer_grads.numpy()
        # 获取输入图片的宽高
        target_size = input_tensor.size(-1), input_tensor.size(-2)
        # Loop over the saliency image from every layer
        # print("layer_grads",type(layer_grads))
        # (1, 256, 13, 13)
        weights = np.mean(grads, axis=(2, 3), keepdims=True)
        # print(weights.shape)  (1, 256, 1, 1)
        # print(activations.shape) (1, 256, 13, 13)
        # 即为每个元素的平均梯度作为该通道的权值,然后计算cam
        weighted_activations = weights * activations
        # print(weighted_activations.shape) (1, 256, 13, 13)
        cam = weighted_activations.sum(axis=1)
        cam[cam < 0] = 0
        scaled = self.scale_cam_image(cam, target_size)
        # print(scaled.shape) (1, 224, 224)
        return scaled
def show_cam_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    # 转化为彩色图片
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        # BGR转化为RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + img
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)

def show_cams_on_image(img: np.ndarray,
                      mask: np.ndarray,
                      mask1: np.ndarray,
                      use_rgb: bool = False,
                      colormap: int = cv2.COLORMAP_JET) -> np.ndarray:
    # 转化为彩色图片
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), colormap)
    if use_rgb:
        # BGR转化为RGB
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    heatmap = np.float32(heatmap) / 255
    heatmap1 = cv2.applyColorMap(np.uint8(255 * mask1), colormap)
    if use_rgb:
        # BGR转化为RGB
        heatmap1 = cv2.cvtColor(heatmap1, cv2.COLOR_BGR2RGB)
    heatmap1 = np.float32(heatmap1) / 255

    cam = heatmap + img + heatmap1
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)






# 加在模型
model = torch.load('./experiment4_data/torch_alex.pth')
model.eval()

target_layers = model.features[-3]
# 预处理,改变图片大小,并且归一化
data_transform = transforms.Compose([transforms.Resize((224, 224)),
                                     transforms.ToTensor()])

img_path = "./experiment4_data/data4/dog.jpg"
# 以RGB格式打开
img = Image.open(img_path).convert('RGB')
# [C, H, W]
img_tensor = data_transform(img)
# expand batch dimension
# 转化为np数组，每个元素取值为0~255，即uint8
img = np.array(img, dtype=np.uint8)
# [C, H, W] -> [N, C, H, W]
# 增加一个维度
input_tensor = torch.unsqueeze(img_tensor, dim=0)
# print(model(input_tensor))
# 创建gradcam实例
cam = GradCAM(model=model, target_layers=target_layers)
# 1表示dog
target_category = [1]
grayscale_cam = cam(input_tensor,target_category)
grayscale_cam = grayscale_cam[0, :]
# 先将原图缩放到0~1


# # 检测cat
# cam = GradCAM(model=model, target_layers=target_layers)
# target_category = [0]
# grayscale_cam1 = cam(input_tensor,target_category)
# grayscale_cam1 = grayscale_cam1[0, :]


visualization = show_cam_on_image(img.astype(dtype=np.float32) / 255.,
                                  grayscale_cam,
                                  use_rgb=True)
plt.imshow(visualization)
plt.show()
