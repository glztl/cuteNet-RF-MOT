import numpy as np
from scipy.optimize import linear_sum_assignment

def associate_detections_to_trackers(detections, trackers):
    # 构造成本矩阵
    cost_matrix = np.zeros((len(trackers), len(detections)))
    for i, tracker in enumerate(trackers):
        for j, detection in enumerate(detections):
            detection_data = np.array([(detection['xmin'] + detection['xmax']) / 2,
                                       (detection['ymin'] + detection['ymax']) / 2])
            cost_matrix[i, j] = np.linalg.norm(tracker.kf.x[:2] - detection_data)

    # 使用匈牙利算法进行数据关联
    track_indices, detection_indices = linear_sum_assignment(cost_matrix)
    return track_indices, detection_indices
