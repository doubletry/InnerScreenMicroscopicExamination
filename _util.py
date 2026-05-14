import socket
from enum import Enum, IntEnum, StrEnum, auto
from pathlib import Path

import cv2
import numpy as np
from hummingbirdai.logger_config import logger
from PySide6.QtCore import QObject, QThread, Signal, Slot


class ResultState(IntEnum):
    PENDING = 0  # 还没结果（默认值）
    OK = 1  # 成功
    NG = 2  # 失败


class ObjectState(Enum):
    DISAPPEARED = auto()  # 完全消失
    APPEARING = auto()  # 正在从消失向出现（过渡态）
    APPEARED = auto()  # 完全出现
    DISAPPEARING = auto()  # 正在从出现向消失（过渡态）


def get_v_channel_brightness(frame):
    """
    计算图像 HSV 空间 V 通道的平均亮度
    :param frame: OpenCV 读取的 BGR 格式图像
    :return: 亮度平均值 (0.0 - 255.0)，如果图像无效则返回 None
    """
    if frame is None:
        return None

    # 1. 将 BGR 转换为 HSV
    # OpenCV 默认 BGR，转换到 HSV 后，通道顺序为 H(0), S(1), V(2)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # 2. 提取 V 通道 (亮度/明度)
    v_channel = hsv[:, :, 2]

    # 3. 计算并返回平均值
    return np.mean(v_channel)


class StateTracker:
    def __init__(self, appear_thresh: int = 3, disappear_thresh: int = 3):
        self.appear_thresh = appear_thresh
        self.disappear_thresh = disappear_thresh

        self.state = ObjectState.DISAPPEARED

        self._appear_count = 0  # 连续调用 appear 的次数
        self._disappear_count = disappear_thresh + 1  # 连续调用 disappear 的次数

    def _enter_appearing_chain(self):
        """
        根据当前连续 appear 次数，决定状态：
        1 ~ appear_thresh-1 ：状态不变
        == appear_thresh    ：进入 APPEARING
        > appear_thresh     ：进入 APPEARED
        """
        if self._appear_count < self.appear_thresh:
            return

        if self._appear_count == self.appear_thresh:
            self.state = ObjectState.APPEARING
        else:
            self.state = ObjectState.APPEARED

    def _enter_disappearing_chain(self):
        """
        根据当前连续 disappear 次数，决定状态：
        1 ~ disappear_thresh-1 ：状态不变
        == disappear_thresh    ：进入 DISAPPEARING
        > disappear_thresh     ：进入 DISAPPEARED
        """
        if self._disappear_count < self.disappear_thresh:
            return

        if self.state == ObjectState.APPEARED:
            if self._disappear_count == self.disappear_thresh:
                self.state = ObjectState.DISAPPEARING
            else:
                self.state = ObjectState.DISAPPEARED

    def appear(self):
        """记录一次 appear 调用，并按规则更新状态"""
        # 方向切换：出现方向被调用，重置消失方向计数
        self._appear_count += 1
        self._disappear_count = 0

        # 不区分当前在哪个方向，统一按 appear 链规则推进
        self._enter_appearing_chain()
        return self.state

    def disappear(self):
        """记录一次 disappear 调用，并按规则更新状态"""
        # 方向切换：消失方向被调用，重置出现方向计数
        self._disappear_count += 1
        self._appear_count = 0

        # 不区分当前在哪个方向，统一按 disappear 链规则推进
        self._enter_disappearing_chain()
        return self.state

    def reset(self, state: ObjectState = ObjectState.DISAPPEARED):
        """重置状态机"""
        self.state = state
        self._appear_count = 0
        self._disappear_count = 0

    def __repr__(self):
        return (
            f"<StateTracker state={self.state.name}, "
            f"appear_count={self._appear_count}, "
            f"disappear_count={self._disappear_count}>"
        )


def in_polygon(point, polygon_points):
    """
    过滤中心点在多边形内部的 YOLO 检测结果

    :param yolo_results: list[np.ndarray]，每个元素形状 (num_boxes, 6)，每个 box: [x1, y1, x2, y2, conf, cls]
    :param polygon_points: list[tuple]，多边形顶点坐标 [(x, y), ...]
    :return: list[np.ndarray]，过滤后的检测结果，格式与输入相同
    """
    point_count = len(polygon_points)
    if point_count == 2:
        # 两点模式 → 转换成矩形 Polygon
        (x1, y1), (x2, y2) = polygon_points
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        return xmin <= point[0] <= xmax and ymin <= point[1] <= ymax

    if point_count < 3:
        return False

    x, y = point
    inside = False
    j = len(polygon_points) - 1
    for i, (xi, yi) in enumerate(polygon_points):
        xj, yj = polygon_points[j]
        intersects = (yi > y) != (yj > y)
        if intersects and yj != yi:
            x_intersect = (xj - xi) * (y - yi) / (yj - yi) + xi
            if x < x_intersect:
                inside = not inside
        j = i

    return inside


def get_package_name():
    if __package__:
        return __package__
    if __spec__ and __spec__.parent:
        return __spec__.parent
    if "." in __name__:
        return __name__.split(".")[0]
    return Path(__file__).parent.name
