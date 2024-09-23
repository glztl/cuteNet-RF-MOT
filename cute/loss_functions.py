import torch
import torch.nn.functional as F

# 自监督噪声抑制损失 (L_SSNS)
def compute_ssns_loss(F_high, F_low, lam):
    """
    Compute self-supervised noise suppression loss.
    """
    high_loss = torch.norm(F_high, p=2) ** 2
    low_loss = lam * (torch.norm(F_low, p=2) ** 2)
    ssns_loss = high_loss - low_loss
    return ssns_loss

# 全局一致性损失 (L_GC)
def compute_gc_loss(F_l, F_l_smoothed):
    """
    Compute global consistency loss using FFT.
    """
    F_l_fft = torch.fft.fft2(F_l)
    F_l_smoothed_fft = torch.fft.fft2(F_l_smoothed)
    gc_loss = torch.norm(F_l_fft - F_l_smoothed_fft, p=2) ** 2
    return gc_loss

# 时域一致性损失 (L_TC)
def compute_tc_loss(F_l_t, F_l_t_next, flow, warp_func):
    """
    Compute temporal consistency loss.
    """
    F_l_t_warped = warp_func(F_l_t, flow)
    tc_loss = torch.norm(F_l_t_next - F_l_t_warped, p=2) ** 2
    return tc_loss

# 追踪损失 (L_track)
def compute_iou(box1, box2):
    """
    Compute Intersection over Union (IoU) for two sets of boxes.
    """
    box1 = convert_to_corners(box1)
    box2 = convert_to_corners(box2)

    inter_x1 = torch.max(box1[:, 0], box2[:, 0])
    inter_y1 = torch.max(box1[:, 1], box2[:, 1])
    inter_x2 = torch.min(box1[:, 2], box2[:, 2])
    inter_y2 = torch.min(box1[:, 3], box2[:, 3])

    inter_area = (inter_x2 - inter_x1).clamp(0) * (inter_y2 - inter_y1).clamp(0)
    box1_area = (box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1])
    box2_area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])
    union_area = box1_area + box2_area - inter_area

    iou = inter_area / union_area
    return iou

def compute_track_loss(pred_boxes, gt_boxes, pred_centers, gt_centers):
    """
    Compute tracking loss using IoU and center point distance.
    """
    iou_loss = 1 - compute_iou(pred_boxes, gt_boxes)
    center_loss = torch.norm(pred_centers - gt_centers, p=2) ** 2
    track_loss = iou_loss + center_loss
    return track_loss

# 总损失函数
def compute_total_loss(track_loss, tc_loss, gc_loss, ssns_loss, alpha, beta, gamma, delta):
    """
    Compute the total loss with weighted sum of different losses.
    """
    total_loss = alpha * track_loss + beta * tc_loss + gamma * gc_loss + delta * ssns_loss
    return total_loss

# 辅助函数：将box从中心坐标转换为角坐标
def convert_to_corners(box):
    """
    Convert boxes from center coordinates to corner coordinates.
    """
    x_center, y_center, width, height = box[:, 0], box[:, 1], box[:, 2], box[:, 3]
    x1 = x_center - width / 2
    y1 = y_center - height / 2
    x2 = x_center + width / 2
    y2 = y_center + height / 2
    return torch.stack([x1, y1, x2, y2], dim=1)