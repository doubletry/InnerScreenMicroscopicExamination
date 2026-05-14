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
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen
from turbojpeg import TurboJPEG

from ._util import (ObjectState, ResultState, StateTracker,
                    get_v_channel_brightness, in_polygon)

IMAGE_MODEL_NAME = ""

MOLD_DETECTION_MODEL_NAME = "上下模检测"
ACTION_MODEL_NAME = "内屏镜检"
NO_FRAME_SEQUENCE_ID = -1

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
    imageReady = Signal(QImage)

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
        self._pending_action_requests = set()
        self._current_action = ResultState.PENDING
        self._is_draining = False
        self._reset_frame_tracking_state()

        self._action_state_tracker = StateTracker()
        self._action_state = None

        self._ok_count = 0
        self._total_count = 0

        self._state_tracker = StateTracker()

        self._action_result_queue = []
        self._mold_status = ResultState.PENDING

    def clear_image_queue(self):

        self.results = OrderedDict()
        self._reset_frame_tracking_state()

    def _reset_frame_tracking_state(self):
        self._upload_image_keys = []
        self._upload_image_list = []
        self._sequence_to_request_id = {}
        self._next_sequence_id = 0
        self._next_emit_sequence_id = 0
        self._last_display_sequence_id = NO_FRAME_SEQUENCE_ID
        self._action_request_id = []
        self._pending_action_requests = set()

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
        self._reset_frame_tracking_state()

    def handle_image(self, image, image_encode=None, request_id=None):
        """
        外部推送一帧图像到ImageClient
        :param image: numpy RGB 图像
        :param image_encode: JPEG编码好的数据（可选）
        :param request_id: 请求ID（可选，不传则生成）
        """
        self._drain_ready_frames()
        pending_frame_count = len(self.results)
        max_pending_frames = self._get_max_pending_frames()
        if pending_frame_count >= max_pending_frames:
            logger.warning(
                f"跳过待处理过多帧，当前待处理={pending_frame_count}, 上限={max_pending_frames}"
            )
            return None

        if request_id is None:
            request_id = secrets.token_hex(4)

        sequence_id = self._next_sequence_id
        self._next_sequence_id += 1

        # Keep a stable frame for upload/state; QImage display detaches its own copy later.
        frame_rgb_array = np.array(image, copy=True, order="C")
        self.results[request_id] = {
            "sequence_id": sequence_id,
            "created_at_ms": time.monotonic() * 1000,
            "image": frame_rgb_array,
        }
        self._sequence_to_request_id[sequence_id] = request_id

        self.upload_image_client.add_input_item(
            {
                "request_id": request_id,
                "image": frame_rgb_array,
                "image_encode": image_encode,
            }
        )
        return request_id

    @Slot(object, object)
    def _on_image_result(self, request_id, resp):
        if request_id not in self.results:
            return

        self.results[request_id]["upload"] = resp
        self._upload_image_count += 1
        key = resp.cache_metas[0].key

        self.mold_detection_client.add_input_item(
            {
                "request_id": request_id,
                "image": None,
                "key": [key],
            }
        )
        self._drain_ready_frames()

    @Slot(object, object)
    def on_mold_detection(self, request_id, resp):
        if request_id not in self.results:
            return

        self.results[request_id]["detection"] = resp
        self._emit_detection_overlay_once(self.results[request_id])
        self._drain_ready_frames()

    def _get_response_timeout_ms(self):
        return self._settings.value("response_timeout_ms", 500, type=int)

    def _get_max_drain_batch(self):
        return max(1, self._settings.value("max_drain_batch", 4, type=int))

    def _get_max_pending_frames(self):
        return max(1, self._settings.value("max_pending_frames", 12, type=int))

    def _is_frame_ready(self, data):
        return "upload" in data and "detection" in data

    def _is_frame_timeout(self, data, now_ms):
        return now_ms - data.get("created_at_ms", now_ms) >= self._get_response_timeout_ms()

    def _drop_sequence(self, sequence_id, request_id):
        self._sequence_to_request_id.pop(sequence_id, None)
        self.results.pop(request_id, None)

    def _get_scaled_area(self, image, settings_key):
        """Scale normalized points from settings into image coordinate tuples."""
        h, w, _ = image.shape
        area_points = self._settings.value(settings_key, [], type=list)
        area = []

        if area_points:
            for point in area_points[0]:
                area.append(
                    (
                        int(point[0] * w),
                        int(point[1] * h),
                    )
                )

        return area

    def _drain_ready_frames(self):
        if self._is_draining:
            return

        self._is_draining = True
        processed = 0
        max_drain_batch = self._get_max_drain_batch()
        try:
            while processed < max_drain_batch:
                sequence_id = self._next_emit_sequence_id
                request_id = self._sequence_to_request_id.get(sequence_id)
                if request_id is None:
                    break

                data = self.results.get(request_id)
                if data is None:
                    self._sequence_to_request_id.pop(sequence_id, None)
                    self._next_emit_sequence_id += 1
                    continue

                now_ms = time.monotonic() * 1000
                if not self._is_frame_ready(data):
                    if self._is_frame_timeout(data, now_ms):
                        logger.warning(
                            f"跳过超时帧 request_id={request_id}, sequence_id={sequence_id}"
                        )
                        self._drop_sequence(sequence_id, request_id)
                        self._next_emit_sequence_id += 1
                        processed += 1
                        continue
                    break

                self._process_ready_frame(request_id, data)
                self._drop_sequence(sequence_id, request_id)
                self._next_emit_sequence_id += 1
                processed += 1
        finally:
            self._is_draining = False

        next_request_id = self._sequence_to_request_id.get(self._next_emit_sequence_id)
        if next_request_id is not None:
            next_data = self.results.get(next_request_id)
            if next_data and self._is_frame_ready(next_data):
                QTimer.singleShot(0, self._drain_ready_frames)

    def _process_ready_frame(self, request_id, data):
        material_area = self._get_scaled_area(data["image"], "material/points")

        box_material = []

        if material_area:
            box_material = [
                material_area[0][0],
                material_area[0][1],
                material_area[1][0],
                material_area[1][1],
            ]

        detection_resp = data["detection"]
        if detection_resp.results:
            for box in detection_resp.results[0].boxes:
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
            upload_resp = data["upload"]
            key = upload_resp.cache_metas[0].key
            if len(self._upload_image_keys) >= self._upload_image_keys_len:
                self._upload_image_keys.pop(0)
                self._upload_image_list.pop(0)

            self._upload_image_keys.append(key)
            self._upload_image_list.append(data["image"].copy())

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
            self._pending_action_requests.add(request_id)
            self._upload_image_keys = []
            self._upload_image_list = []

            self._action_state_tracker.reset()
            self._action_state = None

        self._try_emit(request_id, data)

    def on_action_recognition(self, request_id, resp):
        action_result = ResultState.PENDING
        if resp.results and resp.results[0].label in ["OK", "NG"]:
            action_result = (
                ResultState.OK if resp.results[0].label == "OK" else ResultState.NG
            )
            logger.debug(f"当前动作：{resp.results[0].label}")
        else:
            logger.warning("未获取到动作结果，当前动作结果PENDING")

        self._current_action = action_result

        if request_id in self._action_request_id:
            self._action_request_id.remove(request_id)

        if request_id in self._pending_action_requests:
            self._pending_action_requests.remove(request_id)
            self._mold_status = ResultState.PENDING
            self._total_count += 1
            if action_result == ResultState.OK:
                self._ok_count += 1

            self._log_pass_rate()
            self.resultsReady.emit(
                {
                    "request_id": request_id,
                    "resp": {
                        "result": {
                            "total": self._total_count,
                            "ok": self._ok_count,
                            "current_action": action_result,
                            "current_mold": self._mold_status,
                        }
                    },
                }
            )
            return

        self._action_result_queue.append(action_result)

    def _log_pass_rate(self):
        logger.info(
            f"当前已做{self._total_count}, 一次通过率为{100 * self._ok_count / self._total_count:.2f}"
        )

    def _try_emit(self, request_id, data):

        # 求上下模区域
        mold_area = self._get_scaled_area(data["image"], "mold/points")

        detection_resp = data["detection"]
        if not detection_resp.results:
            logger.warning(
                f"跳过无检测结果帧 request_id={request_id}，请检查检测服务状态"
            )
            return

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
            mold_color = QColor(0, 0, 255)

            if not face_a and not face_c:
                state = self._state_tracker.disappear()
            else:
                state = self._state_tracker.appear()

        elif face_a.y_min < face_c.y_min:
            # 上下模位置错误
            mold_color = QColor(255, 0, 0)
            state = self._state_tracker.appear()

            self._mold_status = ResultState.NG

        else:
            # 上下模位置正确
            mold_color = QColor(0, 255, 0)
            state = self._state_tracker.appear()

            self._mold_status = ResultState.OK

        action_color = self._get_action_color()

        if state == ObjectState.DISAPPEARING:
            self._mold_status = ResultState.PENDING

            if request_id in self._pending_action_requests:
                action_result = ResultState.PENDING
            else:
                action_result = (
                    self._action_result_queue.pop(0)
                    if self._action_result_queue
                    else ResultState.PENDING
                )

                self._total_count += 1
                if action_result == ResultState.OK:
                    self._ok_count += 1
                elif action_result != ResultState.NG:
                    logger.warning(f"未获取到动作结果，当前动作结果PENDING")

                self._log_pass_rate()

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

    def _get_preview_mold_color(self, data):
        """Return QColor for the frame's mold order without mutating state."""
        mold_area = self._get_scaled_area(data["image"], "mold/points")

        detection_resp = data.get("detection")
        if not detection_resp or not detection_resp.results:
            return QColor(0, 0, 255)

        face_a = None
        face_c = None
        for box in detection_resp.results[0].boxes:
            center_x = (box.x_min + box.x_max) / 2
            center_y = (box.y_min + box.y_max) / 2
            if mold_area and not in_polygon((center_x, center_y), mold_area):
                continue

            if box.id == 0:
                face_a = box
            elif box.id == 2:
                face_c = box

        if face_a and face_c and face_a.y_min >= face_c.y_min:
            return QColor(0, 255, 0)
        if face_a and face_c:
            return QColor(255, 0, 0)
        return QColor(0, 0, 255)

    def _get_action_color(self):
        if self._current_action == ResultState.NG:
            return QColor(255, 0, 0)
        if self._current_action == ResultState.OK:
            return QColor(0, 255, 0)
        return QColor(0, 0, 255)

    def _should_emit_image(self, data):
        sequence_id = data.get("sequence_id", NO_FRAME_SEQUENCE_ID)
        if sequence_id < self._last_display_sequence_id:
            return False
        # Raw previews are no longer emitted; one detection overlay per sequence is enough.
        if sequence_id == self._last_display_sequence_id:
            return False
        self._last_display_sequence_id = sequence_id
        return True

    def _emit_detection_overlay_once(self, data, mold_color=None, action_color=None):
        if data.get("annotated_image_emitted"):
            return

        if not self._should_emit_image(data):
            return

        detection_resp = data.get("detection")
        if not detection_resp or not detection_resp.results:
            return

        if mold_color is None:
            mold_color = self._get_preview_mold_color(data)
        if action_color is None:
            action_color = self._get_action_color()

        image = self.draw_detection_on_image(
            data["image"], detection_resp.results[0], mold_color
        )

        if self._current_action == ResultState.OK:
            action_text = "OK"
        elif self._current_action == ResultState.NG:
            action_text = "NG"
        else:
            action_text = "PENDING"

        text = f"内屏镜检撕膜：{action_text}, 明度:{get_v_channel_brightness(data['image']):.1f}"
        image = self.draw_action_on_image(image, (10, 50), text, action_color)

        data["annotated_image_emitted"] = True
        self.imageReady.emit(image)

    def draw_action_on_image(self, image, coord, text, color):
        painter = QPainter(image)
        painter.setRenderHint(QPainter.Antialiasing)
        pen_det = QPen(color, 3)
        font_det = QFont("Arial", 32)
        painter.setFont(font_det)
        painter.setPen(pen_det)

        painter.drawText(QPoint(coord[0], coord[1]), text)
        painter.end()

        return image

    def _image_to_qimage(self, frame_rgb):
        contiguous_frame = np.ascontiguousarray(frame_rgb)
        h, w, ch = contiguous_frame.shape
        bytes_per_line = w * ch
        qimage = QImage(
            contiguous_frame.data, w, h, bytes_per_line, QImage.Format_RGB888
        )
        # This extra copy trades memory bandwidth for safety when frames are reused or deallocated.
        return qimage.copy()

    def draw_detection_on_image(self, frame_rgb, result, color):

        # Step 1: 转为 QImage
        image = self._image_to_qimage(frame_rgb)

        # Step 2: 用 QPainter 绘制
        painter = QPainter(image)
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

        return image
