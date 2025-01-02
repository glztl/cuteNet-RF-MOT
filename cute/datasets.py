import os
import torch
from torch.utils.data import Dataset
from PIL import Image

class MOTDataset(Dataset):
    def __init__(self, image_dir, gt_file, transform=None):
        """
        Args:
            image_dir (str): 图像帧文件夹路径
            gt_file (str): 标注文件路径 (gt.txt)
            transform (callable, optional): 图像预处理变换
        """
        self.image_dir = image_dir
        self.transform = transform

        # 读取标注文件
        self.annotations = []
        with open(gt_file, 'r') as f:
            for line in f:
                try:
                    # 处理空格分隔的数据
                    frame_id, target_id, x_min, y_min, width, height, confidence, class_id, visibility = map(float, line.strip().split())
                    if int(class_id) == 1:  # 仅使用行人类别
                        self.annotations.append({
                            'frame_id': int(frame_id),
                            'target_id': int(target_id),
                            'box': [x_min, y_min, width, height]
                        })
                except ValueError:
                    print(f"Skipping line due to ValueError: {line.strip()}")  # 打印错误行

        # 获取所有图像文件的路径
        self.image_files = sorted(os.listdir(image_dir))

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        # 根据索引加载图像帧和对应的标注
        img_file = self.image_files[idx]
        img_path = os.path.join(self.image_dir, img_file)
        image = Image.open(img_path).convert("RGB")

        # 获取该帧的所有目标标注
        frame_id = idx + 1  # MOT17 标注文件从 1 开始
        gt_boxes = []

        for annotation in self.annotations:
            if annotation['frame_id'] == frame_id:
                x_min, y_min, width, height = annotation['box']
                gt_boxes.append([x_min, y_min, x_min + width, y_min + height])  # 转换为 (x_min, y_min, x_max, y_max) 格式

        gt_boxes = torch.tensor(gt_boxes, dtype=torch.float32)

        # 应用图像预处理（如需要）
        if self.transform:
            image = self.transform(image)

        return image, gt_boxes
