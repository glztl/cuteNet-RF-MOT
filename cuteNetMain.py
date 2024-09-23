import cv2
from det.myModel import YOLOv8Detector
from track.tracker import Tracker
import torch
import torchvision.transforms as transforms
from cute.dfsa import DynamicFrequencySpaceAdaptation


def main(video_path):
    cap = cv2.VideoCapture(video_path)
    detector = YOLOv8Detector()  # 使用 YOLOv8 检测器
    tracker = Tracker()  # 创建追踪器
    dfsa = DynamicFrequencySpaceAdaptation()  # 创建DFSA模块

    transform = transforms.ToTensor()

    frame_id = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # 预处理
        input_image = transform(frame).unsqueeze(0)  # 增加批次维度

        # 使用 YOLOv8 进行目标检测
        detections = detector.detect_objects(frame)

        # 特征提取
        feature_map = dfsa(input_image)

        # 使用追踪器跟踪目标
        tracked_objects = tracker.update(detections)

        # 绘制追踪目标及其 ID
        for obj_id, bbox in tracked_objects:
            x_min, y_min, x_max, y_max = bbox
            cv2.rectangle(frame, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 255, 0), 2)
            cv2.putText(frame, f'ID: {obj_id}', (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        (255, 0, 0), 2)

        cv2.imshow('Tracking', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_id += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    video_path = 'videos/output_video_with_gaussian_rain_tiny.mp4'
    main(video_path)
