import os
import glob
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from yolov5_self import YOLOv5
from PIL import Image
import torch.nn.functional as F


def compute_loss(outputs, targets, num_classes=80):
    pred_cls = outputs[:, :num_classes, :, :]  # 类别预测部分
    pred_conf = outputs[:, num_classes:, :, :]  # 信心预测部分

    loss_cls = torch.tensor(0.0, requires_grad=True).to(outputs.device)  # 初始化为张量
    for i in range(len(targets)):
        target_conf = targets[i][:, 4]  # confidence 假设在最后一列

        if targets[i].size(1) > 5:
            target_cls = targets[i][:, 5].long()  # 假设第6列是类别
        else:
            target_cls = torch.zeros(targets[i].size(0)).long().to(outputs.device)

        # 将目标转换为 one-hot 编码
        target_cls_one_hot = torch.zeros(target_cls.size(0), num_classes).to(target_cls.device)
        target_cls_one_hot[torch.arange(target_cls.size(0)), target_cls] = 1  # 设置相应类别为1

        # 获取类别预测的相关部分
        pred_cls_i = pred_cls[i].view(-1, num_classes)  # 变形为 [N, num_classes]

        # 确保 pred_cls_i 与 target_cls_one_hot 的批次大小相同
        if pred_cls_i.size(0) != target_cls_one_hot.size(0):
            continue  # 跳过不匹配的批次

        # 计算损失
        loss_cls += F.binary_cross_entropy_with_logits(pred_cls_i, target_cls_one_hot.float())

    return loss_cls



class CustomDataset(Dataset):
    def __init__(self, img_dir, label_dir, transform=None):
        self.img_dir = img_dir
        self.label_dir = label_dir
        self.img_paths = glob.glob(os.path.join(img_dir, '*.jpg'))
        self.transform = transform
        self.max_targets = 16  # 设置最大目标数

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label_path = os.path.join(self.label_dir, os.path.basename(img_path).replace('.jpg', '.txt'))

        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)

        with open(label_path, 'r') as f:
            labels = f.readlines()

        targets = []
        for label in labels:
            class_id, x_center, y_center, width, height = map(float, label.strip().split())
            targets.append([class_id, x_center, y_center, width, height])

        targets_tensor = torch.tensor(targets) if targets else torch.zeros((self.max_targets, 5))
        if targets_tensor.size(0) < self.max_targets:
            padding = torch.zeros((self.max_targets - targets_tensor.size(0), 5))
            targets_tensor = torch.cat([targets_tensor, padding], dim=0)

        return image, targets_tensor

def custom_collate(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    targets = [torch.tensor(t) for t in targets]
    return images, targets

transform = transforms.Compose([
    transforms.Resize((640, 640)),
    transforms.ToTensor(),
])

train_dataset = CustomDataset(img_dir='../data/MOT17-RF/train/MOT17-02-FRCNN/img1', label_dir='../data/MOT17-RF/train/MOT17-02-FRCNN/label', transform=transform)
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, collate_fn=custom_collate)

model = YOLOv5(num_classes=80)
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    for images, targets in train_loader:
        optimizer.zero_grad()
        outputs = model(images)

        loss = compute_loss(outputs, targets, num_classes=80)  # 需要传入类别数量
        loss.backward()
        optimizer.step()

    print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

torch.save(model.state_dict(), 'yolov5.pth')
