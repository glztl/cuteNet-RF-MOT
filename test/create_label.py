import os

# 指定图片文件夹路径
image_folder = '../img1/'

# 指定输出文件路径
output_file = '../data/MOT17-RF/train/MOT17-02-FRCNN/label/label.txt'

# 打开文件以写入
with open(output_file, 'w') as f:
    # 遍历从000001到000600的文件
    for i in range(1, 601):
        # 格式化文件名
        file_name = f'{image_folder}{i:06d}.jpg'
        # 写入文件名
        f.write(file_name + '\n')

print(f'已将文件名写入 {output_file}')
