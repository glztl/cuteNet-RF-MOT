import torch
import torch.nn as nn

class Conv(nn.Module):
    """Standard convolution with batch norm and Leaky ReLU."""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(Conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(0.1)
        )

    def forward(self, x):
        return self.conv(x)

class YOLOv5(nn.Module):
    def __init__(self, num_classes=80):
        super(YOLOv5, self).__init__()
        self.backbone = self._build_backbone()
        self.head = self._build_head(num_classes)

    def _build_backbone(self):
        return nn.Sequential(
            Conv(3, 32, kernel_size=6, stride=2, padding=2),  # 1
            Conv(32, 64, kernel_size=3, stride=2),             # 2
            Conv(64, 128),                                     # 3
            Conv(128, 128),                                    # 4
            nn.MaxPool2d(kernel_size=2, stride=2),            # 5
            Conv(128, 256),                                    # 6
            Conv(256, 256),                                    # 7
            nn.MaxPool2d(kernel_size=2, stride=2),            # 8
            Conv(256, 512),                                    # 9
            Conv(512, 512),                                    # 10
            nn.MaxPool2d(kernel_size=2, stride=2),            # 11
            Conv(512, 1024),                                   # 12
            Conv(1024, 1024)                                   # 13
        )

    def _build_head(self, num_classes):
        return nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=1),              # Reduce channels
            nn.Conv2d(512, 1024, kernel_size=3, padding=1),   # Detection layers
            nn.Conv2d(1024, num_classes + 5, kernel_size=1)   # 5 for (x, y, w, h, confidence)
        )

    def forward(self, x):
        features = self.backbone(x)
        outputs = self.head(features)
        return outputs

# 使用示例
if __name__ == "__main__":
    model = YOLOv5(num_classes=80)
    x = torch.randn(1, 3, 640, 640)  # Batch size 1, 3 channels, 640x640 image
    output = model(x)
    print(output.shape)  # Should be [1, num_classes + 5, 20, 20] for example
