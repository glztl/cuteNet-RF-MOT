import torch
import torch.optim as optim
from loss_functions import compute_ssns_loss, compute_gc_loss, compute_tc_loss, compute_track_loss, compute_total_loss
from det.myModel import YOLOv8Detector
from datasets import MOTDataset
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn.functional as F


def compute_high_freq(features):
    kernel = torch.tensor([[0, -1, 0],
                           [-1, 4, -1],
                           [0, -1, 0]], dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    high_freq_features = torch.nn.functional.conv2d(features.unsqueeze(0), kernel, padding=1)
    return high_freq_features.squeeze(0)


def compute_low_freq(features):
    low_freq_features = torch.nn.functional.avg_pool2d(features, kernel_size=2, stride=2)
    return low_freq_features


def smooth_features(features):
    kernel_size = 5
    gaussian_kernel = torch.tensor([[1, 4, 6, 4, 1],
                                    [4, 16, 24, 16, 4],
                                    [6, 24, 36, 24, 6],
                                    [4, 16, 24, 16, 4],
                                    [1, 4, 6, 4, 1]], dtype=torch.float32) / 256.0
    gaussian_kernel = gaussian_kernel.unsqueeze(0).unsqueeze(0)
    smoothed_features = torch.nn.functional.conv2d(features.unsqueeze(0), gaussian_kernel, padding=2)
    return smoothed_features.squeeze(0)


def compute_flow(F_l_t, F_l_t_next):
    flow = F_l_t_next - F_l_t  # Placeholder, 实际情况可能需要使用更复杂的光流算法
    return flow

def warp(image, flow):
    B, C, H, W = image.size()
    grid_x, grid_y = torch.meshgrid(torch.arange(W), torch.arange(H))
    grid = torch.stack((grid_x, grid_y), dim=-1).float()
    grid = grid.unsqueeze(0).expand(B, -1, -1, -1).to(image.device)

    grid += flow.permute(0, 2, 3, 1)
    grid = 2.0 * grid / (W - 1) - 1.0

    warped_image = F.grid_sample(image, grid, mode='bilinear', padding_mode='border')
    return warped_image

# 检查是否有可用的 GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 初始化 YOLOv8 模型
model = YOLOv8Detector(model_path='models/yolov8n.pt')
model.to(device)

# 使用 Adam 优化器
optimizer = optim.Adam(model.model.parameters(), lr=0.001)

# 假设数据集和标注文件路径
image_dir = 'path/to/images'
annotation_file = 'path/to/annotations.json'

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = MOTDataset(image_dir=image_dir, annotation_file=annotation_file, transform=transform)
dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=4)

# 模型训练循环
num_epochs = 10
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0

    for batch_idx, data in enumerate(dataloader):
        images, gt_boxes, gt_centers = data
        images = images.to(device)
        gt_boxes = gt_boxes.to(device)
        gt_centers = gt_centers.to(device)

        outputs = model.forward(images)
        pred_boxes = outputs["detections"]
        pred_centers = outputs.get("centers")

        total_ssns_loss = 0
        total_gc_loss = 0
        total_tc_loss = 0
        num_layers = len(outputs["features"])

        for l in range(num_layers):
            F_l = outputs["features"][l]
            F_l_high = compute_high_freq(F_l)
            F_l_low = compute_low_freq(F_l)
            total_ssns_loss += compute_ssns_loss(F_l_high, F_l_low, lam=0.5)

            F_l_smoothed = smooth_features(F_l)
            total_gc_loss += compute_gc_loss(F_l, F_l_smoothed)

            F_l_t = outputs["features"][l]
            F_l_t_next = outputs["features"][l]  # 假设下一帧特征图
            flow = compute_flow(F_l_t, F_l_t_next)
            total_tc_loss += compute_tc_loss(F_l_t, F_l_t_next, flow, warp_func=warp)

        track_loss = compute_track_loss(pred_boxes, gt_boxes, pred_centers, gt_centers)
        total_loss = compute_total_loss(track_loss, total_tc_loss, total_gc_loss, total_ssns_loss,
                                        alpha=1.0, beta=0.5, gamma=0.5, delta=1.0)

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        running_loss += total_loss.item()
        if batch_idx % 10 == 9:
            print(f'Epoch [{epoch + 1}], Batch [{batch_idx + 1}], Loss: {running_loss / 10:.4f}')
            running_loss = 0.0

    torch.save(model.model.state_dict(), f'models/yolov8_detector_epoch_{epoch + 1}.pt')
    print(f'Model saved for epoch {epoch + 1}.')

torch.save(model.model.state_dict(), 'models/yolov8_detector_final.pt')
print('Final model saved.')