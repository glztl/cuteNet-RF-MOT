import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import MOTDataset  # 导入上面创建的数据加载类
from loss_functions import (compute_ssns_loss, compute_gc_loss, compute_tc_loss,
                            compute_iou, compute_track_loss, compute_total_loss,
                            convert_to_corners, compute_high_freq, compute_low_freq,
                            smooth_features, compute_flow)
from det.myModel import YOLOv8DetectorModule  # 假设你已经定义好该模型

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 相对路径
# 数据集路径
image_dir = '../data/MOT17-RF/train/MOT17-02-FRCNN/img1'  # 图片帧所在文件夹
gt_file = '../data/MOT17-RF/train/MOT17-02-FRCNN/det/det.txt'  # 标注文件路径
# image_dir = '../cute/datasets/coco/images/train2017/'
# gt_file = '../cute/datasets/coco/labels/train2017/'

# 实例化数据集
dataset = MOTDataset(image_dir=image_dir, gt_file=gt_file, transform=transform)

# 使用 DataLoader 进行批处理
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# 初始化 YOLOv8 检测器
model = YOLOv8DetectorModule(model_path='../models/yolov8n.pt')

# 使用 Adam 优化器
optimizer = optim.Adam(model.model.parameters(), lr=0.001)
print({type(model)})

# 模型训练循环
num_epochs = 10
for epoch in range(num_epochs):
    # model.model.train()       # yolov8
    model.train()
    running_loss = 0.0

    for batch_idx, (images, gt_boxes) in enumerate(dataloader):
        images = images.to(device)
        gt_boxes = gt_boxes.to(device)

        # 将图像传入 YOLO 模型进行检测
        detections = []
        pred_centers = []

        for img in images:
            img = img.permute(1, 2, 0).cpu().numpy()  # 转换为 HWC 格式
            detection = model.detect_objects(img)  # 检测
            detections.append(detection)

            # 计算预测中心点
            if detection:
                for box in detection:
                    x_min, y_min, x_max, y_max = box
                    pred_centers.append([(x_min + x_max) / 2, (y_min + y_max) / 2])

        # 转换 detections 为 tensor 格式并移动到设备
        pred_boxes = [torch.tensor(det).to(device) for det in detections if det]  # 只保留有效检测
        if pred_boxes:
            pred_centers = torch.tensor(pred_centers).to(device)

            # 计算损失
            total_ssns_loss = 0
            total_gc_loss = 0
            total_tc_loss = 0
            track_loss = compute_track_loss(pred_boxes, gt_boxes, pred_centers, gt_boxes[:, :2])  # 假设 gt_boxes 包含中心点

            for pred_box in pred_boxes:
                F_l = pred_box  # 获取当前特征
                F_l_high = compute_high_freq(F_l)
                F_l_low = compute_low_freq(F_l)
                total_ssns_loss += compute_ssns_loss(F_l_high, F_l_low, lam=0.5)

                F_l_smoothed = smooth_features(F_l)
                total_gc_loss += compute_gc_loss(F_l, F_l_smoothed)

                # 计算光流和TC损失
                flow = compute_flow(F_l, F_l)  # 使用当前特征作为光流的占位符
                total_tc_loss += compute_tc_loss(F_l, F_l, flow)

            total_loss = compute_total_loss(track_loss, total_tc_loss, total_gc_loss, total_ssns_loss,
                                            alpha=1.0, beta=0.5, gamma=0.5, delta=1.0)

            # Backward and optimize
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            running_loss += total_loss.item()

        if batch_idx % 10 == 9:
            print(f'Epoch [{epoch + 1}], Batch [{batch_idx + 1}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

    # 保存模型
    torch.save(model.state_dict(), f'models/yolov8_detector_epoch_{epoch + 1}.pt')
    print(f'Model saved for epoch {epoch + 1}.')

# 最终模型保存
torch.save(model.state_dict(), 'models/yolov8_detector_final.pt')
print('Final model saved.')
