import numpy as np
from filterpy.kalman import KalmanFilter


class KalmanBoxTracker:
    """
    使用卡尔曼滤波器的跟踪器类
    """

    def __init__(self, obj_id, bbox):
        self.id = obj_id  # 目标ID
        self.kf = self._init_kalman_filter(bbox)  # 初始化卡尔曼滤波器
        self.time_since_update = 0  # 用于记录上次更新后的时间

    def _init_kalman_filter(self, bbox):
        """
        初始化卡尔曼滤波器
        """
        kf = KalmanFilter(dim_x=7, dim_z=4)
        kf.F = np.array([[1, 0, 0, 0, 1, 0, 0],
                         [0, 1, 0, 0, 0, 1, 0],
                         [0, 0, 1, 0, 0, 0, 1],
                         [0, 0, 0, 1, 0, 0, 0],
                         [0, 0, 0, 0, 1, 0, 0],
                         [0, 0, 0, 0, 0, 1, 0],
                         [0, 0, 0, 0, 0, 0, 1]])

        kf.H = np.array([[1, 0, 0, 0, 0, 0, 0],
                         [0, 1, 0, 0, 0, 0, 0],
                         [0, 0, 1, 0, 0, 0, 0],
                         [0, 0, 0, 1, 0, 0, 0]])

        kf.P[4:, 4:] *= 1000.  # 给更高维度较大的不确定性
        kf.P *= 10.
        kf.R[2:, 2:] *= 10.

        # 确保 bbox 是一个二维数组
        kf.x[:4] = np.array(bbox).reshape(4, 1)

        return kf

    def predict(self):
        """
        使用卡尔曼滤波器进行预测
        """
        self.kf.predict()
        self.time_since_update += 1

    def update(self, bbox):
        """
        使用新的检测框更新卡尔曼滤波器
        """
        self.kf.update(bbox)
        self.time_since_update = 0

    def get_state(self):
        """
        获取当前的预测边界框状态
        """
        return self.kf.x[:4].flatten().tolist()

    def is_alive(self):
        """
        判断追踪器是否仍然存活
        """
        return self.time_since_update < 5  # 如果 5 帧未更新，认为该追踪器已失效
