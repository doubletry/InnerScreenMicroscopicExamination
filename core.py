import os
import os.path as osp
import queue
import random
import secrets
import string
import threading
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import cv2
import numpy as np
from hummingbirdai.grpc.core import (ClientBase, DetectionClient,
                                     UploadImageClient,
                                     VideoClassificationClient)
from hummingbirdai.ui import get_path
from loguru import logger
from PySide6.QtCore import QObject, QPoint, QSettings, QThread, QTimer, Signal, Slot
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
    mac = uuid.getnode()
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(mac)))


def compute_iou(box1, box2):
    if not box1 or not box2:
        return 0

    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_w = max(0, inter_x_max - inter_x_min)
    inter_h = max(0, inter_y_max - inter_y_min)
    inter_area = inter_w * inter_h

    area1 = max(0, x1_max - x1_min) * max(0, y1_max - y1_min)
    area2 = max(0, x2_max - x2_min) * max(0, y2_max - y2_min)
    union_area = area1 + area2 - inter_area

    if union_area == 0:
        return 0.0

    return inter_area / union_area


def generate_random_str(length=10):
    chars = string.ascii_letters + string.digits
    return "".join(random.sample(chars, length))


def copy_stable_frame(frame: np.ndarray):
    """立即复制帧，避免持有上游复用缓冲区。"""
    if frame is None:
        return None

    return np.array(frame, copy=True, order="C")


def save_segments(images, roi, root):
    dirname = generate_random_str(12)
    full_dirname = osp.join(root, dirname)
    os.makedirs(full_dirname, exist_ok=True)

    xmin = min(roi[0][0], roi[1][0])
    xmax = max(roi[0][0], roi[1][0])
    ymin = min(roi[0][1], roi[1][1])
    ymax = max(roi[0][1], roi[1][1])

    for i, item in enumerate(images):
        if isinstance(item, dict):
            image = item["image"]
            sequence_index = item.get("sequence_index", i)
        else:
            image = item
            sequence_index = i

        image_path = osp.join(full_dirname, f"{sequence_index:08d}.jpg")
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

    logger.info("退出保存结果线程")


@dataclass
class FrameRecord:
    request_id: str
    sequence_index: int
    image: np.ndarray
    created_at: float = field(default_factory=time.monotonic)
    upload_resp: Any = None
    upload_key: str | None = None
    detection_resp: Any = None
    action_resp: Any = None
    action_requested: bool = False
    action_request_at: float | None = None
    detection_processed: bool = False
    output_ready: bool = False
    output_skipped: bool = False
    output_payload: dict | None = None
    pixmap: QPixmap | None = None
    mold_state: ObjectState | None = None
    mold_color: QColor | None = None


class InnerScreenMicroscopicExaminationClient(QObject):
    """内屏镜检动作检测"""

    resultsReady = Signal(dict)
    imageReady = Signal(QPixmap)

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.results = OrderedDict()
        self.threads = []
        self._settings = settings
        self.client_id = get_machine_unique_id()
        self.image_queue = queue.Queue(1000)

        self._upload_image_keys = []
        self._upload_image_list = []
        self._upload_image_keys_len = 24
        self._action_request_id = set()
        self._current_action = ResultState.PENDING

        self._action_state_tracker = StateTracker()
        self._action_state = None

        self._ok_count = 0
        self._total_count = 0

        self._state_tracker = StateTracker()

        self._action_result_queue = []
        self._mold_status = ResultState.PENDING

        self._lock = threading.RLock()
        self._records_by_request_id: dict[str, FrameRecord] = {}
        self._sequence_to_request_id: dict[int, str] = {}
        self._next_sequence_index = 0
        self._next_detection_sequence = 0
        self._next_output_sequence = 0
        self._timeout_timer: QTimer | None = None

    def clear_image_queue(self):
        with self._lock:
            self.results = OrderedDict()
            self._records_by_request_id.clear()
            self._sequence_to_request_id.clear()
            self._next_sequence_index = 0
            self._next_detection_sequence = 0
            self._next_output_sequence = 0
            self._upload_image_keys = []
            self._upload_image_list = []
            self._action_request_id = set()
            self._action_result_queue = []
            self._action_state_tracker.reset()
            self._action_state = None
            self._state_tracker.reset()
            self._current_action = ResultState.PENDING
            self._mold_status = ResultState.PENDING
            self._ok_count = 0
            self._total_count = 0

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

            self.upload_image_client.resultReady.connect(self._on_image_result)
            self.mold_detection_client.resultReady.connect(self.on_mold_detection)
            self.action_client.resultReady.connect(self.on_action_recognition)

            host = self._settings.value("server_ip", type=str)
            image_port = self._settings.value("image_port", 50050, type=int)
            detection_port = self._settings.value("detection_port", 50051, type=int)
            action_port = self._settings.value("action_port", 50059, type=int)

            if not host or not image_port or not detection_port or not action_port:
                return False

            self.upload_image_client.init_client((host, image_port))
            self.mold_detection_client.init_client((host, detection_port))
            self.action_client.init_client((host, action_port))

            return True
        except Exception:
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
            daemon=True,
        )
        self._save_segments_threading.start()

        self._timeout_timer = QTimer(self)
        self._timeout_timer.setInterval(200)
        self._timeout_timer.timeout.connect(self._on_timeout_tick)
        self._timeout_timer.start()

    def _start_client_thread(self, client_obj: ClientBase):
        t = QThread()
        client_obj.moveToThread(t)
        t.started.connect(client_obj.predict_unary)
        t.start()
        self.threads.append((t, client_obj))

    def stop(self):
        """停止所有子客户端"""
        logger.info(f"停止 {self.__class__.__name__}")
        if self._timeout_timer:
            self._timeout_timer.stop()
            self._timeout_timer.deleteLater()
            self._timeout_timer = None

        for t, client_obj in self.threads:
            client_obj.stop()
        for t, client_obj in self.threads:
            t.quit()
            t.wait()
        self.threads.clear()

        self.image_queue.put(None)
        with self._lock:
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

        image = copy_stable_frame(image)
        with self._lock:
            sequence_index = self._next_sequence_index
            self._next_sequence_index += 1
            record = FrameRecord(request_id, sequence_index, image)
            self._records_by_request_id[request_id] = record
            self._sequence_to_request_id[sequence_index] = request_id
            self.results[request_id] = {"image": image.copy()}

        self.upload_image_client.add_input_item(
            {"request_id": request_id, "image": image.copy(), "image_encode": image_encode}
        )
        self._drain_all()
        return request_id

    @Slot(object, object)
    def _on_image_result(self, request_id, resp):
        key = None
        with self._lock:
            record = self._records_by_request_id.get(request_id)
            if not record or record.output_skipped:
                return
            record.upload_resp = resp
            self.results.setdefault(request_id, {})["upload"] = resp
            if resp.cache_metas:
                key = resp.cache_metas[0].key
                record.upload_key = key

        if key:
            self.mold_detection_client.add_input_item(
                {
                    "request_id": request_id,
                    "image": None,
                    "key": [key],
                }
            )
        self._drain_all()

    @Slot(object, object)
    def on_mold_detection(self, request_id, resp):
        with self._lock:
            record = self._records_by_request_id.get(request_id)
            if not record or record.output_skipped:
                return
            record.detection_resp = resp
            self.results.setdefault(request_id, {})["detection"] = resp
        self._drain_all()

    @Slot(object, object)
    def on_action_recognition(self, request_id, resp):
        with self._lock:
            record = self._records_by_request_id.get(request_id)
            if not record or record.output_skipped:
                return
            record.action_resp = resp
            self.results.setdefault(request_id, {})["action"] = resp
        self._drain_all()

    @Slot()
    def _on_timeout_tick(self):
        self._drain_all()

    def _request_timeout_seconds(self):
        return self._settings.value("request_timeout_seconds", 3.0, type=float)

    def _is_record_timed_out(self, record: FrameRecord):
        return time.monotonic() - record.created_at >= self._request_timeout_seconds()

    def _is_action_timed_out(self, record: FrameRecord):
        start_at = record.action_request_at or record.created_at
        return time.monotonic() - start_at >= self._request_timeout_seconds()

    def _drain_all(self):
        with self._lock:
            self._drain_detection_processing_locked()
            outputs = self._drain_ready_outputs_locked()

        for request_id, data, pixmap in outputs:
            self.resultsReady.emit({"request_id": request_id, "resp": data})
            self.imageReady.emit(pixmap)

    def _record_for_sequence_locked(self, sequence_index: int):
        request_id = self._sequence_to_request_id.get(sequence_index)
        if request_id is None:
            return None
        return self._records_by_request_id.get(request_id)

    def _skip_record_locked(self, record: FrameRecord, reason: str):
        if record.output_skipped:
            return
        record.output_skipped = True
        logger.warning(
            f"丢弃超时帧 request_id={record.request_id}, "
            f"sequence={record.sequence_index}, reason={reason}"
        )
        if record.request_id in self._action_request_id:
            self._action_request_id.discard(record.request_id)

    def _cleanup_record_locked(self, record: FrameRecord):
        self._records_by_request_id.pop(record.request_id, None)
        self._sequence_to_request_id.pop(record.sequence_index, None)
        self.results.pop(record.request_id, None)

    def _drain_detection_processing_locked(self):
        while self._next_detection_sequence < self._next_sequence_index:
            record = self._record_for_sequence_locked(self._next_detection_sequence)
            if record is None:
                self._next_detection_sequence += 1
                continue

            if record.output_skipped:
                self._next_detection_sequence += 1
                continue

            if record.detection_processed:
                self._next_detection_sequence += 1
                continue

            if record.detection_resp is None:
                if self._is_record_timed_out(record):
                    self._skip_record_locked(record, "等待上传/检测结果超时")
                    self._next_detection_sequence += 1
                    continue
                break

            self._process_detection_locked(record)
            self._next_detection_sequence += 1

    def _drain_ready_outputs_locked(self):
        outputs = []
        while self._next_output_sequence < self._next_sequence_index:
            record = self._record_for_sequence_locked(self._next_output_sequence)
            if record is None:
                self._next_output_sequence += 1
                continue

            if record.output_skipped:
                self._cleanup_record_locked(record)
                self._next_output_sequence += 1
                continue

            if not record.detection_processed:
                break

            if record.action_requested and record.action_resp is None:
                if self._is_action_timed_out(record):
                    self._skip_record_locked(record, "等待动作识别结果超时")
                    self._cleanup_record_locked(record)
                    self._next_output_sequence += 1
                    continue
                break

            if not record.output_ready:
                self._finalize_output_locked(record)

            if not record.output_ready:
                break

            outputs.append((record.request_id, record.output_payload, record.pixmap))
            self._cleanup_record_locked(record)
            self._next_output_sequence += 1
        return outputs

    def _material_area_for_image(self, image):
        h, w, _ = image.shape
        material_area_points = self._settings.value("material/points", [], type=list)
        material_area = []

        if material_area_points:
            for point in material_area_points[0]:
                material_area.append((int(point[0] * w), int(point[1] * h)))
        return material_area

    def _mold_area_for_image(self, image):
        h, w, _ = image.shape
        mold_area_points = self._settings.value("mold/points", [], type=list)
        mold_area = []

        if mold_area_points:
            for point in mold_area_points[0]:
                mold_area.append((int(point[0] * w), int(point[1] * h)))
        return mold_area

    def _process_detection_locked(self, record: FrameRecord):
        material_area = self._material_area_for_image(record.image)
        box_material = []
        if len(material_area) >= 2:
            box_material = [
                material_area[0][0],
                material_area[0][1],
                material_area[1][0],
                material_area[1][1],
            ]

        detection_results = record.detection_resp.results if record.detection_resp else []
        if detection_results:
            for box in detection_results[0].boxes:
                if box.id not in [6, 7]:
                    continue

                box_coord = [box.x_min, box.y_min, box.x_max, box.y_max]
                if compute_iou(box_coord, box_material) < 0.3:
                    continue

                if box.id == 7:
                    self._action_state = self._action_state_tracker.appear()
                elif box.id == 6:
                    self._action_state = self._action_state_tracker.disappear()

                break

        if self._action_state == ObjectState.APPEARED and record.upload_key:
            self._append_action_clip_locked(record)

        elif self._action_state == ObjectState.DISAPPEARING:
            self._submit_action_clip_locked(record, material_area)

        self._process_mold_status_locked(record)
        record.detection_processed = True

    def _append_action_clip_locked(self, record: FrameRecord):
        if len(self._upload_image_keys) >= self._upload_image_keys_len:
            self._upload_image_keys.pop(0)
            self._upload_image_list.pop(0)

        self._upload_image_keys.append(record.upload_key)
        self._upload_image_list.append(
            {
                "sequence_index": record.sequence_index,
                "image": record.image.copy(),
            }
        )

    def _submit_action_clip_locked(self, record: FrameRecord, material_area):
        sequence_keys = list(self._upload_image_keys)
        clip_images = [
            {"sequence_index": item["sequence_index"], "image": item["image"].copy()}
            for item in self._upload_image_list
        ]

        if not sequence_keys:
            self._upload_image_keys = []
            self._upload_image_list = []
            self._action_state_tracker.reset()
            self._action_state = None
            return

        request = {
            "request_id": record.request_id,
            "sequences_keys": [sequence_keys],
            "sequences_rois": [[list(material_area) for _ in sequence_keys]],
        }

        self.action_client.add_input_item(request)
        record.action_requested = True
        record.action_request_at = time.monotonic()
        self._action_request_id.add(record.request_id)

        save_clip = self._settings.value("save_clip", False, type=bool)
        if save_clip and clip_images and len(material_area) >= 2:
            try:
                self.image_queue.put_nowait(
                    {
                        "images": clip_images,
                        "roi": list(material_area),
                    }
                )
            except queue.Full:
                logger.exception("保存图片失败")

        self._upload_image_keys = []
        self._upload_image_list = []
        self._action_state_tracker.reset()
        self._action_state = None

    def _process_mold_status_locked(self, record: FrameRecord):
        mold_area = self._mold_area_for_image(record.image)
        detection_results = record.detection_resp.results if record.detection_resp else []

        face_a = None
        face_c = None
        if detection_results and mold_area:
            for box in detection_results[0].boxes:
                center_x = (box.x_min + box.x_max) / 2
                center_y = (box.y_min + box.y_max) / 2
                if not in_polygon((center_x, center_y), mold_area):
                    continue

                if box.id == 0:
                    face_a = box
                elif box.id == 2:
                    face_c = box

        if not face_a or not face_c:
            record.mold_color = QColor(0, 0, 255)
            if not face_a and not face_c:
                record.mold_state = self._state_tracker.disappear()
            else:
                record.mold_state = self._state_tracker.appear()

        elif face_a.y_min < face_c.y_min:
            record.mold_color = QColor(255, 0, 0)
            record.mold_state = self._state_tracker.appear()
            self._mold_status = ResultState.NG

        else:
            record.mold_color = QColor(0, 255, 0)
            record.mold_state = self._state_tracker.appear()
            self._mold_status = ResultState.OK

    def _finalize_output_locked(self, record: FrameRecord):
        if record.action_requested and record.action_resp is None:
            return

        data = self.results.pop(record.request_id, {})
        data["image"] = record.image.copy()
        data["upload"] = record.upload_resp
        data["detection"] = record.detection_resp
        if record.action_resp is not None:
            data["action"] = record.action_resp

        detection_results = record.detection_resp.results if record.detection_resp else []
        detection_result = detection_results[0] if detection_results else SimpleNamespace(names=[], boxes=[])
        pixmap = self.draw_detection_on_image(
            record.image.copy(), detection_result, record.mold_color or QColor(0, 0, 255)
        )

        data["drawn"] = pixmap

        if record.action_requested:
            action_resp = record.action_resp
            self._action_request_id.discard(record.request_id)

            if action_resp and action_resp.results and action_resp.results[0].label in ["OK", "NG"]:
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

        if record.mold_state == ObjectState.DISAPPEARING:
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
                logger.warning("未获取到动作结果，当前动作结果None")

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

        if self._current_action == ResultState.OK:
            action_text = "OK"
        elif self._current_action == ResultState.NG:
            action_text = "NG"
        else:
            action_text = "PENDING"

        text = f"内屏镜检撕膜：{action_text}, 明度:{get_v_channel_brightness(data['image']):.1f}"
        pixmap = self.draw_action_on_pixmap(pixmap, (10, 50), text, color)

        record.output_payload = data
        record.pixmap = pixmap
        record.output_ready = True

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
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        image_bytes = frame_rgb.tobytes()
        qimg = QImage(image_bytes, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

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

            label = names[box.id] if box.id < len(names) else str(box.id)
            label = f"{label} {box.confidence:.2f}"
            painter.drawText(QPoint(x1, y1 - 5), label)

        painter.end()

        return pixmap
