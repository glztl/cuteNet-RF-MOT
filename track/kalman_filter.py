import numpy as np
from filterpy.kalman import KalmanFilter

class KalmanTracker:
    def __init__(self):
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        self.kf.x = np.array([0, 0, 0, 0])  # 初始状态 [x, y, vx, vy]
        self.kf.P *= 1000.  # 初始不确定性
        self.kf.F = np.array([[1, 0, 1, 0],
                              [0, 1, 0, 1],
                              [0, 0, 1, 0],
                              [0, 0, 0, 1]])  # 状态转移矩阵
        self.kf.H = np.array([[1, 0, 0, 0],
                              [0, 1, 0, 0]])  # 观测矩阵
        self.kf.R = np.array([[10, 0],
                              [0, 10]])  # 观测噪声
        self.kf.Q = np.eye(4)  # 过程噪声

    def predict(self):
        self.kf.predict()
        return self.kf.x

    def update(self, z):
        self.kf.update(z)
        return self.kf.x
