import socket
from enum import Enum, auto
from pathlib import Path

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot
from shapely.geometry import Point, Polygon

from hummingbirdai.logger_config import logger


class ObjectState(Enum):
    DISAPPEARED = auto()  # 完全消失
    APPEARING = auto()  # 正在从消失向出现（过渡态）
    APPEARED = auto()  # 完全出现
    DISAPPEARING = auto()  # 正在从出现向消失（过渡态）


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
    if len(polygon_points) == 2:
        # 两点模式 → 转换成矩形 Polygon
        (x1, y1), (x2, y2) = polygon_points
        xmin, xmax = min(x1, x2), max(x1, x2)
        ymin, ymax = min(y1, y2), max(y1, y2)
        polygon_points = [(xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax)]

    poly = Polygon(polygon_points)
    center_point = Point(point[0], point[1])
    return poly.contains(center_point)


class TcpRequestWorker(QObject):
    """
    在单独线程中执行 TCP 请求的 Worker 类。
    """

    # 定义信号：
    # finished: 请求成功时发出，携带接收到的字节数据
    # error: 请求失败时发出，携带发生的异常对象
    finished = Signal(bytes)
    error = Signal(object)

    def __init__(self, address, data, timeout=1.0):
        super().__init__()
        self._address = address
        self._data = data
        self._timeout = timeout

    @Slot()
    def run(self):
        """
        执行实际的 TCP 请求逻辑。
        此方法将在 QThread 启动后被调用。
        """
        recv_chunks = []
        try:
            # 创建并连接
            with socket.create_connection(self._address, timeout=self._timeout) as sock:
                # 发送所有数据
                sock.sendall(self._data)
                sock.settimeout(self._timeout)
                try:
                    chunk = sock.recv(1024)
                    recv_chunks.append(chunk)
                except socket.timeout as e:
                    # 超时就认为对方不再发了
                    logger.exception("tcp请求异常")
                    self.error.emit(e)
                    return

            # 请求成功，发出 finished 信号
            self.finished.emit(b"".join(recv_chunks))
        except Exception as e:
            # 请求失败，发出 error 信号
            logger.exception("tcp请求异常")
            self.error.emit(e)


def tcp_request_async(
    parent, address, data, timeout=1.0, on_success=None, on_error=None
):
    """
    异步 TCP 客户端请求。
    此函数会启动一个新线程来执行 TCP 请求，不会阻塞 UI。

    :param address: (host, port)，例如 ("127.0.0.1", 5000)
    :param data: 要发送的字节数据（bytes）
    :param timeout: socket 超时时间（秒）
    :param on_success: 请求成功时调用的回调函数，接收一个参数：收到的 bytes
    :param on_error: 请求失败时调用的回调函数，接收一个参数：异常对象
    :return: 启动的 QThread 实例，你可以选择性地保留它以便后续管理（如等待线程结束），
             但通常为了不阻塞 UI，你不需要直接操作它。
    """
    # 1. 创建一个 QThread 对象
    thread = QThread(parent)
    # 2. 创建一个 Worker 对象，将请求参数传递给它
    worker = TcpRequestWorker(address, data, timeout)
    # 3. 将 Worker 移动到新线程中
    worker.moveToThread(thread)

    # 4. 连接信号和槽：
    # 当线程启动时，调用 worker 的 run 方法
    thread.started.connect(worker.run)

    # 当 worker 完成（成功或失败）时，执行相应的回调函数
    if on_success:
        worker.finished.connect(on_success)
    if on_error:
        worker.error.connect(on_error)

    # 5. 连接清理信号：
    # 无论 worker 成功还是失败，都让线程退出
    worker.finished.connect(thread.quit)
    worker.error.connect(thread.quit)
    # 当线程退出时，安全地删除 worker 和 thread 对象
    thread.finished.connect(worker.deleteLater)
    thread.finished.connect(thread.deleteLater)

    # 6. 启动线程
    thread.start()

    return thread, worker


def tcp_request(address, data, timeout=1.0):
    """
    简单 TCP 客户端：
    - 连接到 address = (host, port)
    - 发送 data (bytes)
    - 接收服务器返回的数据（直到对方关闭连接或超时）
    - 关闭连接
    - 返回收到的 bytes

    :param address: (host, port)，例如 ("127.0.0.1", 5000)
    :param data: 要发送的字节数据（bytes）
    :param timeout: socket 超时时间（秒）
    :return: 服务器返回的所有 bytes
    """
    recv_chunks = []

    # 创建并连接
    with socket.create_connection(address, timeout=timeout) as sock:
        # 发送所有数据
        sock.sendall(data)

        # 告诉服务器：我这边已经发完了（半关闭写端）
        try:
            sock.shutdown(socket.SHUT_WR)
        except OSError:
            # 某些平台/情况可能不支持 shutdown，忽略即可
            pass

        # 接收数据，直到对方关闭或超时
        sock.settimeout(timeout)
        while True:
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                # 超时就认为对方不再发了
                break

            if not chunk:
                # 对方关闭连接
                break

            recv_chunks.append(chunk)

    return b"".join(recv_chunks)


def get_package_name():
    if __package__:
        return __package__
    if __spec__ and __spec__.parent:
        return __spec__.parent
    if "." in __name__:
        return __name__.split(".")[0]
    return Path(__file__).parent.name
