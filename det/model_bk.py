import torch
from ultralytics import YOLO


class YOLOv8Detector:
    def __init__(self, model_path='models/yolov8n.pt'):
        # 初始化 YOLOv8 模型
        self.model = YOLO(model_path)

    def detect_objects(self, frame):
        # 使用 YOLOv8 进行目标检测
        results = self.model(frame)

        detections = []
        for box in results[0].boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0]
            detections.append([x_min.item(), y_min.item(), x_max.item(), y_max.item()])

        return detections, results[0].features  # 返回检测结果和特征图

    def forward(self, frames):
        # 直接处理一批帧，用于训练时
        results = self.model(frames)

        detections = []
        for box in results[0].boxes:
            x_min, y_min, x_max, y_max = box.xyxy[0]
            detections.append([x_min.item(), y_min.item(), x_max.item(), y_max.item()])

        return {"detections": detections, "features": results[0].features}