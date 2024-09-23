import torch
import torch.nn.functional as F

def compute_ssns_loss(F_high, F_low, lam):
    # L_SSNS = ||F_high||_2^2 - λ * ||F_low||_2^2
    high_loss = torch.norm(F_high, p=2) ** 2
    low_loss = lam * (torch.norm(F_low, p=2) ** 2)
    ssns_loss = high_loss - low_loss
    return ssns_loss

def compute_gc_loss(F_l, F_l_smoothed):
    # L_GC = ||FFT(F_l) - FFT(F_l_smoothed)||_2^2
    F_l_fft = torch.fft.fft2(F_l)
    F_l_smoothed_fft = torch.fft.fft2(F_l_smoothed)
    gc_loss = torch.norm(F_l_fft - F_l_smoothed_fft, p=2) ** 2
    return gc_loss

def compute_tc_loss(F_l_t, F_l_t_next, flow):
    # 将t时刻的特征图F_l^t通过flow进行warp
    F_l_t_warped = warp(F_l_t, flow)
    # L_TC = ||F_l^(t+1) - warp(F_l^t, flow)||_2^2
    tc_loss = torch.norm(F_l_t_next - F_l_t_warped, p=2) ** 2
    return tc_loss

def warp(feature_map, flow):
    # 假设flow是一个二维光流字段 (u, v)，需要对feature_map进行warp
    # 使用双线性插值进行warping
    n, c, h, w = feature_map.size()
    grid = create_flow_grid(flow, h, w)
    warped_feature = F.grid_sample(feature_map, grid, align_corners=True)
    return warped_feature

def compute_iou(box1, box2):
    # box1, box2 格式为 [x_center, y_center, width, height]
    # 将中心点坐标和宽高转换为左上角和右下角坐标
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
    # L_track = (1 - IoU) + ||C_pred - C_gt||_2^2
    iou_loss = 1 - compute_iou(pred_boxes, gt_boxes)
    center_loss = torch.norm(pred_centers - gt_centers, p=2) ** 2
    track_loss = iou_loss + center_loss
    return track_loss

def compute_total_loss(track_loss, tc_loss, gc_loss, ssns_loss, alpha, beta, gamma, delta):
    # L_Total = αL_track + βL_TC + γL_GC + δL_SSNS
    total_loss = alpha * track_loss + beta * tc_loss + gamma * gc_loss + delta * ssns_loss
    return total_loss


