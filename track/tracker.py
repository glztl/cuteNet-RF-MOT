import numpy as np
from scipy.optimize import linear_sum_assignment
from utils.kalman_tracker import KalmanBoxTracker

class Tracker:
    def __init__(self):
        self.trackers = []  # 保存现有的卡尔曼滤波追踪器
        self.next_id = 0    # 为每个新目标分配唯一ID

    def update(self, detections):
        """
        更新追踪器状态。对每一帧进行检测，并与现有追踪器进行匹配。
        """
        # 1. 预测步骤：对每个卡尔曼滤波追踪器执行预测
        for tracker in self.trackers:
            tracker.predict()

        # 2. 匈牙利算法进行匹配
        if len(self.trackers) > 0 and len(detections) > 0:
            # 计算检测框和追踪器的IOU代价矩阵
            cost_matrix = self._iou_cost_matrix(detections)

            # 使用匈牙利算法找到最佳匹配
            row_ind, col_ind = linear_sum_assignment(cost_matrix)

            assigned_detections = set()  # 已分配的检测
            assigned_trackers = set()    # 已分配的追踪器

            for r, c in zip(row_ind, col_ind):
                # 只匹配 IOU 大于 0.5 的检测结果
                if cost_matrix[r, c] < 0.5:
                    self.trackers[c].update(detections[r])  # 更新对应追踪器
                    assigned_detections.add(r)
                    assigned_trackers.add(c)

            # 3. 为未分配的检测框创建新的追踪器
            for i, det in enumerate(detections):
                if i not in assigned_detections:
                    self._create_tracker(det)

        else:
            # 没有追踪器或检测结果，创建新的追踪器
            for det in detections:
                self._create_tracker(det)

        # 4. 清理失效的追踪器
        self.trackers = [t for t in self.trackers if t.is_alive()]

        # 返回所有追踪器的状态
        return [(tracker.id, tracker.get_state()) for tracker in self.trackers]

    def _create_tracker(self, detection):
        """
        创建一个新的卡尔曼滤波追踪器。
        """
        tracker = KalmanBoxTracker(self.next_id, detection)
        self.trackers.append(tracker)
        self.next_id += 1

    def _iou_cost_matrix(self, detections):
        """
        计算 IOU 代价矩阵。
        """
        cost_matrix = np.zeros((len(detections), len(self.trackers)))
        for i, det in enumerate(detections):
            for j, tracker in enumerate(self.trackers):
                cost_matrix[i, j] = 1 - self._iou(det, tracker.get_state())  # 1 - IOU 作为代价
        return cost_matrix

    def _iou(self, boxA, boxB):
        """
        计算两个边界框之间的 IOU（Intersection over Union）。
        """
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[2], boxB[2])
        yB = min(boxA[3], boxB[3])
        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
        boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
        iou = interArea / float(boxAArea + boxBArea - interArea)
        return iou
