import cv2
import torchvision
import torch
from matplotlib import pyplot as plt
import numpy as np
import random
import torchvision.transforms as transforms


# 定义函数，随机从0-end的一个序列中抽取size个不同的数
def random_num(size, end):
    range_ls = [i for i in range(end)]
    num_ls = []
    for i in range(size):
        num = random.choice(range_ls)
        range_ls.remove(num)
        num_ls.append(num)
    return num_ls


# 对特征图应用傅里叶变换并提取高频和低频部分
def split_freq(v_channel, radius=10):
    # 对二维特征图应用傅里叶变换
    fft = np.fft.fft2(v_channel.cpu().numpy())
    fft_shift = np.fft.fftshift(fft)

    # 创建用于掩蔽低频和高频的遮罩
    rows, cols = v_channel.shape
    crow, ccol = rows // 2, cols // 2  # 找到频谱的中心
    mask_low = np.zeros((rows, cols), np.uint8)
    mask_high = np.ones((rows, cols), np.uint8)

    # 设置低频遮罩为1 (保留中心部分)，其余部分为0
    mask_low[crow - radius:crow + radius, ccol - radius:ccol + radius] = 1
    # 高频遮罩是全1减去低频遮罩，即只保留外围部分
    mask_high -= mask_low

    # 分别获取低频和高频频谱
    low_freq = fft_shift * mask_low
    high_freq = fft_shift * mask_high

    # 将频谱转换回空间域
    low_freq_img = np.fft.ifft2(np.fft.ifftshift(low_freq))
    high_freq_img = np.fft.ifft2(np.fft.ifftshift(high_freq))

    # 取实部作为结果
    return np.abs(low_freq_img), np.abs(high_freq_img)


path = "img/img.png"
transformss = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Resize((224, 224)),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

# 注意如果有中文路径需要先解码，最好不要用中文
img = cv2.imread(path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# 转换维度
img = transformss(img).unsqueeze(0)

model = torchvision.models.resnet50(weights=torchvision.models.ResNet50_Weights.DEFAULT)
new_model = torchvision.models._utils.IntermediateLayerGetter(model, {'layer1': '1', 'layer2': '2', "layer3": "3"})
out = new_model(img)

tensor_ls = [(k, v) for k, v in out.items()]

# 选取layer2的输出
v = tensor_ls[1][1]
v = v.data.squeeze(0)  # 去掉 batch 维度

# 打印特征图形状
print(v.shape)  # torch.Size([512, 28, 28])

# 随机选取25个通道的特征图
channel_num = random_num(25, v.shape[0])

plt.figure(figsize=(10, 10))
for index, channel in enumerate(channel_num):
    ax = plt.subplot(5, 5, index + 1)

    # 获取原始特征图
    original_feature_map = v[channel, :, :]

    # 将特征图分为低频和高频
    low_freq, high_freq = split_freq(original_feature_map, radius=5)

    # 可视化低频或高频特征 (这里展示低频特征)
    plt.imshow(low_freq, cmap='gray')
    ax.set_title(f'Low Freq {channel}')
    ax.axis('off')

plt.savefig("low_freq_feature.jpg", dpi=300)

plt.figure(figsize=(10, 10))
for index, channel in enumerate(channel_num):
    ax = plt.subplot(5, 5, index + 1)

    # 获取原始特征图
    original_feature_map = v[channel, :, :]

    # 将特征图分为低频和高频
    low_freq, high_freq = split_freq(original_feature_map, radius=5)

    # 可视化高频特征
    plt.imshow(high_freq, cmap='gray')
    ax.set_title(f'High Freq {channel}')
    ax.axis('off')

plt.savefig("high_freq_feature.jpg", dpi=300)
