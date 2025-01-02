import os

# 设置输入和输出目录
det_file_path = '../data/MOT17-RF/train/MOT17-02-FRCNN/det/det.txt'  # MOT17 的 det.txt 文件
output_dir = '../data/MOT17-RF/train/MOT17-02-FRCNN/label'  # YOLO 标签输出目录

# 创建输出目录
os.makedirs(output_dir, exist_ok=True)

# 图像的宽度和高度（根据实际图像进行修改）
img_width = 1920
img_height = 1080

# 读取 det.txt 文件
with open(det_file_path, 'r') as f:
    for line in f:
        parts = line.strip().split(',')
        frame_id = int(parts[0])  # 帧编号
        bbox_left = float(parts[2])
        bbox_top = float(parts[3])
        bbox_width = float(parts[4])
        bbox_height = float(parts[5])

        # 计算 YOLO 格式
        x_center = (bbox_left + bbox_width / 2) / img_width
        y_center = (bbox_top + bbox_height / 2) / img_height
        width = bbox_width / img_width
        height = bbox_height / img_height

        # 创建标签文件
        yolo_label_file = os.path.join(output_dir, f'{frame_id:06d}.txt')

        # 假设所有对象都为同一类别 0
        with open(yolo_label_file, 'a') as yolo_file:
            yolo_file.write(f'0 {x_center} {y_center} {width} {height}\n')

print("转换完成！")
