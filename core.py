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
from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np
from hummingbirdai.grpc.core import (ClientBase, DetectionClient,
                                     UploadImageClient,
                                     VideoClassificationClient)
from loguru import logger
from PySide6.QtCore import QObject, QPoint, QSettings, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap

from ._util import (DEFAULT_ACTION_CLIP_LENGTH, DEFAULT_REQUEST_TIMEOUT_MS,
                    ObjectState, ResultState, StateTracker,
                    get_v_channel_brightness, in_polygon)

IMAGE_MODEL_NAME = ""
MOLD_DETECTION_MODEL_NAME = "上下模检测"
ACTION_MODEL_NAME = "内屏镜检"
SAVE_THREAD_JOIN_TIMEOUT_SECONDS = 2

current_file_path = os.path.abspath(__file__)
current_dir = os.path.dirname(current_file_path)


@dataclass
class FrameRecord:
    request_id: str
    sequence_index: int
    image: np.ndarray
    created_at: float
    upload_resp: Any = None
    upload_key: str | None = None
    detection_resp: Any = None
    detection_processed: bool = False
    action_resp: Any = None
    action_requested: bool = False


def copy_stable_frame(image, max_attempts=5):
    """
    Copy a frame only after two consecutive snapshots are identical.

    Some video producers reuse and overwrite the same ndarray buffer from another
    thread. A single copy can catch that buffer mid-write, so retry until the
    copied shape, dtype, and pixels match the previous snapshot.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be a positive integer (at least 1)")
    previous = np.array(image, copy=True, order="C")
    # The first copy above counts as attempt 1; the loop performs the remaining
    # attempts until two adjacent snapshots are identical.
    for _ in range(max_attempts - 1):
        current = np.array(image, copy=True, order="C")
        if (
            current.shape == previous.shape
            and current.dtype == previous.dtype
            and np.array_equal(current, previous)
        ):
            return current
        previous = current
    return previous


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


def _clip_directory_name(sequence_indices):
    random_suffix = generate_random_str(6)
    if sequence_indices is not None and len(sequence_indices) > 0:
        return (
            f"seq{sequence_indices[0]:06d}_"
            f"seq{sequence_indices[-1]:06d}_{random_suffix}"
        )
    return f"{int(time.time() * 1000)}_{random_suffix}"


def save_segments(images, roi, root, sequence_indices=None):
    if len(roi) < 2:
        logger.warning("保存片段失败：未配置有效物料区域")
        return

    first_image = images[0] if images else None
    if first_image is None:
        logger.warning("保存片段失败：图像为空")
        return
    if first_image.ndim < 2:
        logger.warning("保存片段失败：图像尺寸无效")
        return

    dirname = _clip_directory_name(sequence_indices)
    full_dirname = osp.join(root, dirname)
    os.makedirs(full_dirname, exist_ok=True)

    height, width = first_image.shape[:2]
    xmin = max(0, min(roi[0][0], roi[1][0]))
    xmax = min(width, max(roi[0][0], roi[1][0]))
    ymin = max(0, min(roi[0][1], roi[1][1]))
    ymax = min(height, max(roi[0][1], roi[1][1]))
    if xmin >= xmax or ymin >= ymax:
        logger.warning("保存片段失败：物料区域无效或为空")
        return

    for i, image in enumerate(images):
        sequence_index = i
        if sequence_indices is not None and i < len(sequence_indices):
            sequence_index = sequence_indices[i]
        image_path = osp.join(full_dirname, f"seq{sequence_index:06d}.jpg")
        image_bgr = cv2.cvtColor(np.ascontiguousarray(image), cv2.COLOR_RGB2BGR)
        crop = np.ascontiguousarray(image_bgr[ymin:ymax, xmin:xmax])
        cv2.imwrite(image_path, crop)

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
        sequence_indices = data.get("sequence_indices")
        logger.debug(f"获取到分割图片，共{len(images)}张")
        save_segments(images, roi, root, sequence_indices)

    logger.info("退出保存结果线程")


class InnerScreenMicroscopicExaminationClient(QObject):
    """内屏镜检动作检测"""

    resultsReady = Signal(dict)
    imageReady = Signal(QPixmap)

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self.results: OrderedDict[str, FrameRecord] = OrderedDict()
        self.threads = []
        self._settings = settings
        self.client_id = get_machine_unique_id()
        self.image_queue = queue.Queue(1000)
        self._save_segments_threading = None

        self._sequence_to_request_id: OrderedDict[int, str] = OrderedDict()
        self._next_sequence_index = 0
        self._next_detection_sequence = 0
        self._next_emit_sequence = 0
        self._current_clip: list[dict] = []
        self._action_request_id = []
        self._current_action = ResultState.PENDING
        self._action_state_tracker = StateTracker()
        self._action_state = None
        self._ok_count = 0
        self._total_count = 0
        self._state_tracker = StateTracker()
        self._action_result_queue = []
        self._mold_status = ResultState.PENDING
        self._state_lock = threading.RLock()

    def clear_image_queue(self):
        self._clear_pending_image_queue(keep_stop_signal=True)
        with self._state_lock:
            self._reset_runtime_state(reset_counts=True)

    def _clear_pending_image_queue(self, keep_stop_signal=False):
        while True:
            try:
                item = self.image_queue.get_nowait()
            except queue.Empty:
                break
            if item is None and keep_stop_signal:
                self.image_queue.put(None)
                break

    def _reset_runtime_state(self, reset_counts=False):
        self.results = OrderedDict()
        self._sequence_to_request_id = OrderedDict()
        self._next_sequence_index = 0
        self._next_detection_sequence = 0
        self._next_emit_sequence = 0
        self._current_clip = []
        self._action_request_id = []
        self._current_action = ResultState.PENDING
        self._action_state_tracker.reset()
        self._action_state = None
        self._state_tracker.reset()
        self._action_result_queue = []
        self._mold_status = ResultState.PENDING
        if reset_counts:
            self._ok_count = 0
            self._total_count = 0

    def _action_clip_length(self):
        return max(
            1,
            self._settings.value(
                "action_clip_length", DEFAULT_ACTION_CLIP_LENGTH, type=int
            ),
        )

    def _request_timeout_seconds(self):
        timeout_ms = self._settings.value(
            "request_timeout_ms", DEFAULT_REQUEST_TIMEOUT_MS, type=int
        )
        return max(1, timeout_ms) / 1000.0

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
        logger.info(f"启动 {self.__class__.__name__}")
        self._start_client_thread(self.upload_image_client)
        self._start_client_thread(self.mold_detection_client)
        self._start_client_thread(self.action_client)

        self._clear_pending_image_queue()
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
        logger.info(f"停止 {self.__class__.__name__}")
        for _, client_obj in self.threads:
            client_obj.stop()
        for t, _ in self.threads:
            t.quit()
            t.wait()
        self.threads.clear()

        self.image_queue.put(None)
        if self._save_segments_threading and self._save_segments_threading.is_alive():
            self._save_segments_threading.join(
                timeout=SAVE_THREAD_JOIN_TIMEOUT_SECONDS
            )
            # Do not block plugin shutdown indefinitely; if saving is still busy,
            # leave the worker to finish its current item and warn for diagnostics.
            if self._save_segments_threading.is_alive():
                logger.warning("保存结果线程未在超时时间内退出")
        self._save_segments_threading = None
        with self._state_lock:
            self._reset_runtime_state(reset_counts=False)

    def handle_image(self, image, image_encode=None, request_id=None):
        with self._state_lock:
            if request_id is None:
                request_id = secrets.token_hex(4)

            # 即使调用方已经拷贝过，这里再用 ndarray.copy() 做一次受控的、独立缓冲的拷贝，
            # 保证 FrameRecord 持有的图像与上传客户端持有的图像彼此完全独立，
            # 避免任意一方在后台线程修改/编码图像时影响到显示与保存。
            source = copy_stable_frame(image)
            record_frame = source.copy()
            upload_frame = source.copy()

            sequence_index = self._next_sequence_index
            self._next_sequence_index += 1

            self.results[request_id] = FrameRecord(
                request_id=request_id,
                sequence_index=sequence_index,
                image=record_frame,
                created_at=time.time(),
            )
            self._sequence_to_request_id[sequence_index] = request_id

            self.upload_image_client.add_input_item(
                {
                    "request_id": request_id,
                    "image": upload_frame,
                    "image_encode": image_encode,
                }
            )
            return request_id

    @Slot(object, object)
    def _on_image_result(self, request_id, resp):
        with self._state_lock:
            record = self.results.get(request_id)
            if not record:
                logger.warning(f"收到未知或已丢弃的上传结果：{request_id}")
                return

            key = self._extract_upload_key(resp)
            if key is None:
                logger.warning(f"上传结果缺少图像key：{request_id}")
                self._drop_record(record, "upload key missing")
                self._drain_ready_detection_results()
                self._drain_ready_outputs()
                return

            record.upload_resp = resp
            record.upload_key = key
            self.mold_detection_client.add_input_item(
                {"request_id": request_id, "image": None, "key": [key]}
            )
            self._drain_ready_detection_results()
            self._drain_ready_outputs()

    @Slot(object, object)
    def on_mold_detection(self, request_id, resp):
        with self._state_lock:
            record = self.results.get(request_id)
            if not record:
                logger.warning(f"收到未知或已丢弃的检测结果：{request_id}")
                return

            record.detection_resp = resp
            self._drain_ready_detection_results()
            self._drain_ready_outputs()

    def _extract_upload_key(self, resp):
        cache_metas = getattr(resp, "cache_metas", None)
        if not cache_metas:
            return None
        return getattr(cache_metas[0], "key", None)

    def _drain_ready_detection_results(self):
        while self._next_detection_sequence < self._next_sequence_index:
            request_id = self._sequence_to_request_id.get(self._next_detection_sequence)
            if request_id is None:
                self._next_detection_sequence += 1
                continue

            record = self.results.get(request_id)
            if record is None:
                self._sequence_to_request_id.pop(self._next_detection_sequence, None)
                self._next_detection_sequence += 1
                continue

            if record.detection_resp is None:
                if time.time() - record.created_at >= self._request_timeout_seconds():
                    self._drop_record(record, "request timeout")
                    self._next_detection_sequence += 1
                    continue
                break

            self._process_ordered_detection(record)
            self._next_detection_sequence += 1

    def _drop_record(self, record: FrameRecord, reason: str):
        logger.warning(
            f"丢弃超时/无效帧 request_id={record.request_id}, "
            f"sequence={record.sequence_index}, reason={reason}"
        )
        self.results.pop(record.request_id, None)
        self._sequence_to_request_id.pop(record.sequence_index, None)
        if record.request_id in self._action_request_id:
            self._action_request_id.remove(record.request_id)

    def _process_ordered_detection(self, record: FrameRecord):
        material_area = self._build_area(record.image, "material")
        box_material = self._area_to_box(material_area)
        detection_result = self._first_result(record.detection_resp)

        if detection_result and box_material:
            for box in detection_result.boxes:
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

        if self._action_state == ObjectState.APPEARED:
            self._append_action_clip(record, material_area)
        elif self._action_state == ObjectState.DISAPPEARING:
            self._submit_action_clip(record, material_area)

        record.detection_processed = True
        self._drain_ready_outputs()

    def _append_action_clip(self, record: FrameRecord, material_area):
        if record.upload_key is None:
            return

        # 立即对 record.image 做独立拷贝。FrameRecord 在 _try_emit 之后会被丢弃，
        # 而片段需要在动作识别返回 / 保存线程消费完之前一直存在，必须断开与 FrameRecord 的共享。
        self._current_clip.append(
            {
                "sequence_index": record.sequence_index,
                "key": record.upload_key,
                "image": np.ascontiguousarray(record.image).copy(),
                "roi": list(material_area),
            }
        )
        clip_length = self._action_clip_length()
        if len(self._current_clip) > clip_length:
            self._current_clip = self._current_clip[-clip_length:]

    def _submit_action_clip(self, record: FrameRecord, material_area):
        # _current_clip 已按 sequence_index 顺序追加（由 _drain_ready_detection_results 保证），
        # 这里再次按 sequence_index 排序作为防御，确保保存的 0.jpg, 1.jpg, ... 与采集顺序一致。
        clip = sorted(self._current_clip, key=lambda item: item["sequence_index"])
        keys = [item["key"] for item in clip]
        rois = [item["roi"] or list(material_area) for item in clip]

        if keys:
            request = {
                "request_id": record.request_id,
                "sequences_keys": [keys],
                "sequences_rois": [rois],
            }
            self.action_client.add_input_item(request)
            record.action_requested = True
            self._action_request_id.append(record.request_id)

            if self._settings.value("save_clip", False, type=bool):
                try:
                    # 每张图像都用独立缓冲传给保存线程，避免后续帧的写入产生拼接。
                    saver_images = [
                        np.ascontiguousarray(item["image"]).copy() for item in clip
                    ]
                    self.image_queue.put_nowait(
                        {
                            "images": saver_images,
                            "sequence_indices": [
                                item["sequence_index"] for item in clip
                            ],
                            "roi": list(material_area),
                        }
                    )
                except queue.Full:
                    logger.exception("保存图片失败")
        else:
            logger.warning(f"动作片段为空，跳过动作识别：{record.request_id}")

        self._current_clip = []
        self._action_state_tracker.reset()
        self._action_state = None

    def on_action_recognition(self, request_id, resp):
        with self._state_lock:
            record = self.results.get(request_id)
            if not record:
                logger.warning(f"收到未知或已丢弃的动作结果：{request_id}")
                return

            record.action_resp = resp
            self._drain_ready_outputs()

    def _is_ready_to_emit(self, record: FrameRecord):
        if record is None or not record.detection_processed:
            return False

        if record.action_requested and record.action_resp is None:
            return False

        return True

    def _drain_ready_outputs(self):
        while self._next_emit_sequence < self._next_sequence_index:
            request_id = self._sequence_to_request_id.get(self._next_emit_sequence)
            if request_id is None:
                self._next_emit_sequence += 1
                continue

            record = self.results.get(request_id)
            if record is None:
                self._sequence_to_request_id.pop(self._next_emit_sequence, None)
                self._next_emit_sequence += 1
                continue

            if not self._is_ready_to_emit(record):
                break

            self._emit_record(record)
            self._next_emit_sequence += 1

    def _emit_record(self, record: FrameRecord):
        request_id = record.request_id
        mold_area = self._build_area(record.image, "mold")
        detection_result = self._first_result(record.detection_resp)
        face_a, face_c = self._find_mold_faces(detection_result, mold_area)

        if not face_a or not face_c:
            color = QColor(0, 0, 255)
            if not face_a and not face_c:
                state = self._state_tracker.disappear()
            else:
                state = self._state_tracker.appear()
        elif face_a.y_min < face_c.y_min:
            color = QColor(255, 0, 0)
            state = self._state_tracker.appear()
            self._mold_status = ResultState.NG
        else:
            color = QColor(0, 255, 0)
            state = self._state_tracker.appear()
            self._mold_status = ResultState.OK

        pixmap = self.draw_detection_on_image(record.image, detection_result, color)
        data = {
            "image": np.ascontiguousarray(record.image).copy(),
            "upload": record.upload_resp,
            "detection": record.detection_resp,
            "drawn": pixmap,
        }

        if request_id in self._action_request_id:
            self._action_request_id.remove(request_id)
            action_resp = record.action_resp
            if action_resp and action_resp.results:
                label = action_resp.results[0].label
                if label in ["OK", "NG"]:
                    self._current_action = (
                        ResultState.OK if label == "OK" else ResultState.NG
                    )
                    logger.debug(f"当前动作：{label}")
                    self._action_result_queue.append(self._current_action)
            data["action"] = action_resp

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
                logger.warning("未获取到动作结果，当前动作结果None")

            logger.info(
                f"当前已做{self._total_count}, "
                f"一次通过率为{100 * self._ok_count / self._total_count:.2f}"
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

        text = (
            f"内屏镜检撕膜：{action_text}, "
            f"明度:{get_v_channel_brightness(record.image):.1f}"
        )
        pixmap = self.draw_action_on_pixmap(pixmap, (10, 50), text, color)
        self.imageReady.emit(pixmap)

        self.results.pop(request_id, None)
        self._sequence_to_request_id.pop(record.sequence_index, None)

    def _build_area(self, image, group):
        h, w, _ = image.shape
        area_points = self._settings.value(f"{group}/points", [], type=list)
        area = []
        if area_points:
            for point in area_points[0]:
                area.append((int(point[0] * w), int(point[1] * h)))
        return area

    def _area_to_box(self, area):
        if len(area) < 2:
            return []
        return [area[0][0], area[0][1], area[1][0], area[1][1]]

    def _first_result(self, resp):
        results = getattr(resp, "results", None)
        if not results:
            return None
        return results[0]

    def _find_mold_faces(self, detection_result, mold_area):
        face_a = None
        face_c = None
        if not detection_result or len(mold_area) < 2:
            return face_a, face_c

        for box in detection_result.boxes:
            center_x = (box.x_min + box.x_max) / 2
            center_y = (box.y_min + box.y_max) / 2
            if not in_polygon((center_x, center_y), mold_area):
                continue

            if box.id == 0:
                face_a = box
            elif box.id == 2:
                face_c = box
        return face_a, face_c

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
        # 用 tobytes() 生成一个不可变的 Python bytes 对象作为 QImage 的底层缓冲，
        # 这样 QImage / QPixmap 完全脱离原始 numpy 数组的内存。即便后续帧覆盖了
        # 原始缓冲，本次绘制出来的画面也不会被污染（避免显示画面拼接）。
        contiguous = np.ascontiguousarray(frame_rgb)
        h, w, ch = contiguous.shape
        bytes_per_line = ch * w
        # 显式保留 bytes 引用，避免 QImage 仍在使用时被 GC 回收。
        buffer = contiguous.tobytes()
        qimg = QImage(buffer, w, h, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(qimg)

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        pen_det = QPen(color, 3)
        font_det = QFont("Arial", 32)
        painter.setFont(font_det)

        if result:
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
