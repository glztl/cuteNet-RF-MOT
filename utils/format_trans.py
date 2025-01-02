import pandas as pd

# 读取原始数据
data = pd.read_csv('../data/MOT17-RF/train/MOT17-02-FRCNN/gt/gt.txt', header=None)

# 提取需要的列
data['x_center'] = (data[2] + data[4]) / 2
data['y_center'] = (data[3] + data[5]) / 2
data['width'] = data[4] - data[2]
data['height'] = data[5] - data[3]

# 假设 class_id 在第二列
data['class_id'] = data[1]

# 标准化到 [0, 1]
image_width, image_height = 1280, 720  # 设定图像宽高
data['x_center'] /= image_width
data['y_center'] /= image_height
data['width'] /= image_width
data['height'] /= image_height

# 选择需要的列并保存为 YOLO 格式
yolo_format = data[['class_id', 'x_center', 'y_center', 'width', 'height']]
yolo_format.to_csv('yolo_format.txt', header=False, index=False, sep=' ')
