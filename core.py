import os
import os.path as osp
import queue
import random
import re
import secrets
import string
import threading
import time
import uuid
from collections import OrderedDict

import av
import cv2
import grpc
import numpy as np
from dateutil.parser import parse
from hummingbirdai.grpc import base_pb2
from hummingbirdai.grpc.core import (ClientBase, DetectionClient,
                                     UploadImageClient,
                                     VideoClassificationClient)
from hummingbirdai.multimedia import FrameSampler
from hummingbirdai.ui import get_path
from loguru import logger
from PySide6.QtCore import (QObject, QPoint, QSettings, Qt, QThread, QTimer,
                            Signal, Slot)
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
from turbojpeg import TurboJPEG

from ._util import (ObjectState, ResultState, StateTracker,
                    get_v_channel_brightness, in_polygon)

IMAGE_MODEL_NAME = ""

MOLD_DETECTION_MODEL_NAME = "上下模检测"
ACTION_MODEL_NAME = "内屏镜检"

TURBO_JPEG_DLL = get_path("dlls/libturbojpeg.dll")

jpeg = TurboJPEG(TURBO_JPEG_DLL)
current_file_path = os.path.abspath(__file__)  # 当前文件的绝对路径
current_dir = os.path.dirname(current_file_path)  # 当前文件所在目录


def get_machine_unique_id():
    # 基于 MAC 地址 + 时间戳生成 UUID
    # 如果只需要完全固定的ID，可以直接用 MAC 地址
    mac = uuid.getnode()
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(mac)))


def compute_iou(box1, box2):
    """
    计算两个矩形框的 IoU (Intersection over Union)

    参数:
        box1, box2: tuple 或 list，格式为 (x1, y1, x2, y2)
            (x1, y1) 为左上角坐标
            (x2, y2) 为右下角坐标

    返回:
        iou: float，两个框的 IoU 值
    """

    if not box1 or not box2:
        return 0

    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    # 计算交集矩形的左上角和右下角坐标
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    # 计算交集区域的宽和高
    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)

    # 交集面积
    inter_area = inter_w * inter_h

    # 各自面积
    area1 = max(0, x1_max - x1_min) * max(0, y1_max - y1_min)
    area2 = max(0, x2_max - x2_min) * max(0, y2_max - y2_min)

    # 并集面积
    union_area = area1 + area2 - inter_area

    # 防止除零
    if union_area == 0:
        return 0.0

    iou = inter_area / union_area
    return iou


def generate_random_str(length=10):
    # 字符集：字母 + 数字
    chars = string.ascii_letters + string.digits
    # random.sample 返回一个长度为 length 的唯一字符列表
    return "".join(random.sample(chars, length))


def save_segments(images, roi, root):

    dirname = generate_random_str(12)
    full_dirname = osp.join(root, dirname)
    os.makedirs(full_dirname, exist_ok=True)

    xmin = min(roi[0][0], roi[1][0])
    xmax = max(roi[0][0], roi[1][0])
    ymin = min(roi[0][1], roi[1][1])
    ymax = max(roi[0][1], roi[1][1])

    for i, image in enumerate(images):
        image_path = osp.join(full_dirname, f"{i}.jpg")

        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)[ymin:ymax, xmin:xmax]
        cv2.imwrite(image_path, image)

    logger.info(f"保存分割图片成功，共{len(images)}张，保存到{full_dirname}")


def backend_save_segments(image_queue: queue.Queue, root):
    logger.info(f"开始保存分割图片到{root}")
    while True:
        try:
            data = image_queue.get(timeout=1)
        except queue.Empty:
            continue

        if data is None:
            break

        images = data["images"]
        roi = data["roi"]
        logger.debug(f"获取到分割图片，共{len(images)}张")

        save_segments(images, roi, root)

    logger.info(f"退出保存结果线程")


class InnerScreenMicroscopicExaminationClient(QObject):
    """内屏镜检动作检测"""

    resultsReady = Signal(dict)
    imageReady = Signal(QPixmap)

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.results = OrderedDict()
        self.threads = []  # [(thread, client_obj), ...]
        # 三个子客户端
        self._settings = settings
        self.client_id = get_machine_unique_id()
        self.image_queue = queue.Queue(1000)

        self._frame_index = -1
        self.frame_sampler = FrameSampler()

        self._upload_image_count = 0
        self._upload_image_keys = []
        self._upload_image_list = []
        self._upload_image_keys_len = 24
        self._action_request_id = []
        self._current_action = ResultState.PENDING

        self._action_state_tracker = StateTracker()
        self._action_state = None

        self._ok_count = 0
        self._total_count = 0

        self._state_tracker = StateTracker()

        self._action_result_queue = []
        self._mold_status = ResultState.PENDING

    def clear_image_queue(self):

        self.results = OrderedDict()

    def init_client(self):
        try:

            self.upload_image_client = UploadImageClient(
                self.client_id, IMAGE_MODEL_NAME
            )

            self.mold_detection_client = DetectionClient(
                self.client_id, MOLD_DETECTION_MODEL_NAME
            )

            self.action_client = VideoClassificationClient(
                self.client_id, ACTION_MODEL_NAME
            )

            # 绑定结果信号

            self.upload_image_client.resultReady.connect(self._on_image_result)
            self.mold_detection_client.resultReady.connect(self.on_mold_detection)
            self.action_client.resultReady.connect(self.on_action_recognition)

            host = self._settings.value("server_ip", type=str)
            image_port = self._settings.value("image_port", 50050, type=int)
            detection_port = self._settings.value("detection_port", 50051, type=int)
            action_port = self._settings.value("action_port", 50059, type=int)

            if not host or not image_port or not detection_port or not action_port:
                return False

            # 初始化连接

            self.upload_image_client.init_client((host, image_port))
            self.mold_detection_client.init_client((host, detection_port))
            self.action_client.init_client((host, action_port))

            return True
        except:
            logger.exception("初始化异常")
            return False

    def start(self):
        """启动三个客户端的连接和预测线程"""
        logger.info(f"启动 {self.__class__.__name__}")

        self._start_client_thread(self.upload_image_client)
        self._start_client_thread(self.mold_detection_client)
        self._start_client_thread(self.action_client)

        self._save_segments_threading = threading.Thread(
            target=backend_save_segments,
            args=(self.image_queue, osp.join(current_dir, "history")),
        )
        self._save_segments_threading.start()

    def _start_client_thread(self, client_obj: ClientBase):
        t = QThread()
        client_obj.moveToThread(t)
        t.started.connect(client_obj.predict_unary)
        t.start()
        self.threads.append((t, client_obj))

    def stop(self):
        """停止所有子客户端"""
        logger.info(f"停止 {self.__class__.__name__}")
        for t, client_obj in self.threads:
            t: QThread
            client_obj: ClientBase
            client_obj.stop()
        for t, client_obj in self.threads:
            t.quit()
            t.wait()
        self.threads.clear()

        self.image_queue.put(None)
        self._upload_image_keys = []
        self._upload_image_list = []

    def handle_image(self, image, image_encode=None, request_id=None):
        """
        外部推送一帧图像到ImageClient
        :param image: numpy RGB 图像
        :param image_encode: JPEG编码好的数据（可选）
        :param request_id: 请求ID（可选，不传则生成）
        """
        if request_id is None:
            request_id = secrets.token_hex(4)

        self.results.setdefault(request_id, {})["image"] = image

        self.upload_image_client.add_input_item(
            {"request_id": request_id, "image": image, "image_encode": image_encode}
        )
        return request_id

    @Slot(object, object)
    def _on_image_result(self, request_id, resp):
        self.results.setdefault(request_id, {})["upload"] = resp
        self._upload_image_count += 1
        key = resp.cache_metas[0].key

        self.mold_detection_client.add_input_item(
            {
                "request_id": request_id,
                "image": None,
                "key": [key],
            }
        )

    @Slot(object, object)
    def on_mold_detection(self, request_id, resp):
        self.results.setdefault(request_id, {})["detection"] = resp

        h, w, _ = self.results[request_id]["image"].shape
        material_area_points = self._settings.value(
            "material/points",
            [],
            type=list,
        )

        material_area = []

        if material_area_points:
            for point in material_area_points[0]:
                material_area.append(
                    (
                        int(point[0] * w),
                        int(point[1] * h),
                    )
                )

        box_material = []

        if material_area:
            box_material = [
                material_area[0][0],
                material_area[0][1],
                material_area[1][0],
                material_area[1][1],
            ]

        if resp.results:
            for box in resp.results[0].boxes:
                if box.id not in [6, 7]:  # 如果不是物料的两个框，则跳过
                    continue

                box_coord = [box.x_min, box.y_min, box.x_max, box.y_max]
                if (
                    compute_iou(box_coord, box_material) < 0.3
                ):  # 如果物料框和检测框的iou小于0.3，则跳过
                    continue

                if box.id == 7:  # material undo
                    self._action_state = self._action_state_tracker.appear()
                elif box.id == 6:
                    self._action_state = self._action_state_tracker.disappear()

                break

        if self._action_state == ObjectState.APPEARED:
            upload_resp = self.results[request_id]["upload"]
            key = upload_resp.cache_metas[0].key
            if len(self._upload_image_keys) >= self._upload_image_keys_len:
                self._upload_image_keys.pop(0)
                self._upload_image_list.pop(0)

            self._upload_image_keys.append(key)
            self._upload_image_list.append(self.results[request_id]["image"])

        elif self._action_state == ObjectState.DISAPPEARING:
            request = {
                "request_id": request_id,
                "sequences_keys": [self._upload_image_keys],
                "sequences_rois": [[material_area for _ in self._upload_image_keys]],
            }

            self.action_client.add_input_item(request)

            save_clip = self._settings.value("save_clip", False, type=bool)

            if save_clip:
                try:
                    self.image_queue.put_nowait(
                        {
                            "images": self._upload_image_list,
                            "roi": material_area,
                        }
                    )
                except queue.Full:
                    logger.exception("保存图片失败")

            self._action_request_id.append(request_id)
            self._upload_image_keys = []
            self._upload_image_list = []

            self._action_state_tracker.reset()
            self._action_state = None

        self._try_emit(request_id)

    def on_action_recognition(self, request_id, resp):
        self.results.setdefault(request_id, {})["action"] = resp
        self._try_emit(request_id)

    def _try_emit(self, request_id):

        if (
            request_id in self._action_request_id
            and "action" not in self.results[request_id]
        ):
            return

        if "detection" not in self.results[request_id]:
            return

        # 求上下模区域
        h, w, _ = self.results[request_id]["image"].shape
        mold_area_points = self._settings.value("mold/points", [], type=list)

        mold_area = []

        if mold_area_points:
            for point in mold_area_points[0]:
                mold_area.append(
                    (
                        int(point[0] * w),
                        int(point[1] * h),
                    )
                )

        data = self.results.pop(request_id)

        detection_resp = data["detection"]

        # 过滤上下模区域内的物体
        face_a = None
        face_c = None
        for box in detection_resp.results[0].boxes:
            center_x = (box.x_min + box.x_max) / 2
            center_y = (box.y_min + box.y_max) / 2
            if not in_polygon((center_x, center_y), mold_area):
                continue

            if box.id == 0:
                face_a = box
            elif box.id == 2:
                face_c = box

        if not face_a or not face_c:
            # 上下模不存在
            color = QColor(0, 0, 255)

            if not face_a and not face_c:
                state = self._state_tracker.disappear()
            else:
                state = self._state_tracker.appear()

        elif face_a.y_min < face_c.y_min:
            # 上下模位置错误
            color = QColor(255, 0, 0)
            state = self._state_tracker.appear()

            self._mold_status = ResultState.NG

        else:
            # 上下模位置正确
            color = QColor(0, 255, 0)
            state = self._state_tracker.appear()

            self._mold_status = ResultState.OK

        pixmap = self.draw_detection_on_image(
            data["image"], data["detection"].results[0], color
        )

        data["drawn"] = pixmap

        # 如果有检测动作
        if request_id in self._action_request_id:

            action_resp = data["action"]
            self._action_request_id.remove(request_id)

            if not action_resp.results:
                return

            if action_resp.results[0].label in ["OK", "NG"]:
                self._current_action = (
                    ResultState.OK
                    if action_resp.results[0].label == "OK"
                    else ResultState.NG
                )
                logger.debug(f"当前动作：{action_resp.results[0].label}")

                self._action_result_queue.append(self._current_action)

        if self._current_action == ResultState.NG:
            color = QColor(255, 0, 0)

        elif self._current_action == ResultState.OK:
            color = QColor(0, 255, 0)

        else:
            color = QColor(0, 0, 255)

        if state == ObjectState.DISAPPEARING:
            action_result = (
                self._action_result_queue.pop(0)
                if self._action_result_queue
                else ResultState.PENDING
            )
            self._mold_status = ResultState.PENDING

            if action_result == ResultState.OK:
                self._ok_count += 1
                self._total_count += 1
            elif action_result == ResultState.NG:
                self._total_count += 1
            else:
                self._total_count += 1
                logger.warning(f"未获取到动作结果，当前动作结果None")

            logger.info(
                f"当前已做{self._total_count}, 一次通过率为{100 * self._ok_count / self._total_count:.2f}"
            )
            self._state_tracker.reset()
            self._current_action = ResultState.PENDING

        else:
            action_result = (
                self._action_result_queue[0]
                if self._action_result_queue
                else ResultState.PENDING
            )

        data["result"] = {
            "total": self._total_count,
            "ok": self._ok_count,
            "current_action": action_result,
            "current_mold": self._mold_status,
        }

        self.resultsReady.emit({"request_id": request_id, "resp": data})

        if self._current_action == ResultState.OK:
            action_text = "OK"
        elif self._current_action == ResultState.NG:
            action_text = "NG"
        else:
            action_text = "PENDING"

        text = f"内屏镜检撕膜：{action_text}, 明度:{get_v_channel_brightness(data['image']):.1f}"

        pixmap = self.draw_action_on_pixmap(pixmap, (10, 50), text, color)

        self.imageReady.emit(pixmap)

    def draw_action_on_pixmap(self, pixmap, coord, text, color):
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen_det = QPen(color, 3)
        font_det = QFont("Arial", 32)
        painter.setFont(font_det)
        painter.setPen(pen_det)

        painter.drawText(QPoint(coord[0], coord[1]), text)
        painter.end()

        return pixmap

    def draw_detection_on_image(self, frame_rgb, result, color):

        # Step 1: 转为 QImage
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)

        # Step 2: 创建 QPixmap
        pixmap = QPixmap.fromImage(qimg)

        # Step 3: 用 QPainter 绘制
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # 绘制 detection

        pen_det = QPen(color, 3)
        font_det = QFont("Arial", 32)
        painter.setFont(font_det)

        names = result.names

        for box in result.boxes:
            x1, y1, x2, y2 = (
                int(box.x_min),
                int(box.y_min),
                int(box.x_max),
                int(box.y_max),
            )
            painter.setPen(pen_det)
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)

            label = names[box.id]
            label = f"{label} {box.confidence:.2f}"
            painter.drawText(QPoint(x1, y1 - 5), label)

        painter.end()

        return pixmap
