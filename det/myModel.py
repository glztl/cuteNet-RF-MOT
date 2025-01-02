import torch
import cv2
import numpy as np
from ultralytics import YOLO  # 直接使用YOLOv8

class YOLOv8DetectorModule:
    def __init__(self, model_path):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = YOLO(model_path).to(self.device)

    def preprocess_image(self, image):
        img = cv2.resize(image, (640, 640))
        img = img[:, :, ::-1]  # BGR to RGB
        img = img / 255.0  # Normalize to [0, 1]
        return img

    def detect_objects(self, image):
        img = self.preprocess_image(image)
        img_tensor = torch.tensor(img).permute(2, 0, 1).unsqueeze(0).to(self.device)
        with torch.no_grad():
            results = self.model(img_tensor)
        return results[0].boxes




