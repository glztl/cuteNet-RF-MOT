import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
import json


class MOTDataset(Dataset):
    def __init__(self, image_dir, annotation_file, transform=None):
        """
        Args:
            image_dir (str): 图像文件夹的路径
            annotation_file (str): 标签文件路径 (JSON, COCO 或 YOLO 格式等)
            transform (callable, optional): 图像预处理变换
        """
        self.image_dir = image_dir
        self.annotation_file = annotation_file
        self.transform = transform

        # 加载标注文件，例如 JSON 格式
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

    def __len__(self):
        # 返回数据集大小
        return len(self.annotations)

    def __getitem__(self, idx):
        # 根据索引加载图像和标签
        annotation = self.annotations[idx]
        img_path = os.path.join(self.image_dir, annotation['image'])
        image = Image.open(img_path).convert("RGB")

        # 读取边界框 (gt_boxes) 和中心点 (gt_centers)
        gt_boxes = torch.tensor(annotation['boxes'])  # 每个box格式: [x_min, y_min, x_max, y_max]
        gt_centers = torch.tensor(annotation['centers'])  # 中心点格式: [x_center, y_center]

        # 应用图像变换（如需要）
        if self.transform:
            image = self.transform(image)

        return image, gt_boxes, gt_centers


# 假设数据集和标注文件路径
image_dir = 'path/to/images'  # 图像文件夹
annotation_file = 'path/to/annotations.json'  # 标注文件

# 实例化数据集
dataset = MOTDataset(image_dir=image_dir, annotation_file=annotation_file)

# 使用 DataLoader 进行批处理
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)