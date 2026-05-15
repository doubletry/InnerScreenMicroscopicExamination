import os
import os.path as osp
import queue
import secrets
import threading
import time
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np
from hummingbirdai.grpc.core import (
    ClientBase,
    DetectionClient,
    UploadImageClient,
    VideoClassificationClient,
)
from loguru import logger
from PySide6.QtCore import QObject, QPoint, QSettings, Qt, QThread, QTimer, Signal, Slot
from PySide6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap

from ._util import ObjectState, ResultState, StateTracker, get_v_channel_brightness, in_polygon

IMAGE_MODEL_NAME = ""
MOLD_DETECTION_MODEL_NAME = "上下模检测"
ACTION_MODEL_NAME = "内屏镜检"
MATERIAL_EMPTY_BOX_ID = 6
MATERIAL_PRESENT_BOX_ID = 7

current_dir = os.path.dirname(os.path.abspath(__file__))


def get_machine_unique_id():
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, str(uuid.getnode())))


def copy_stable_frame(frame: np.ndarray) -> np.ndarray:
    """Return an owned C-contiguous RGB copy to avoid async buffer reuse issues.

    复制一份连续内存的 RGB 图像，避免上游复用帧缓冲区导致异步读写串帧。
    """
    return np.array(frame, copy=True, order="C")


def compute_iou(box1, box2):
    if not box1 or not box2:
        return 0.0

    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2
    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)
    inter_area = max(0, inter_x_max - inter_x_min) * max(0, inter_y_max - inter_y_min)
    area1 = max(0, x1_max - x1_min) * max(0, y1_max - y1_min)
    area2 = max(0, x2_max - x2_min) * max(0, y2_max - y2_min)
    union_area = area1 + area2 - inter_area
    return inter_area / union_area if union_area else 0.0


@dataclass
class FrameRecord:
    """Complete context for one frame in the asynchronous pipeline.

    Upload, detection, and action recognition are returned asynchronously by
    different threads/services. sequence_index is the primary ordering key used
    to keep the input frame, service responses, and final display result for the
    same frame together before ordered processing.

    单帧在异步流水线中的完整上下文。
    上传、检测和动作识别分别由不同线程/服务异步返回。该对象以
    sequence_index 为主键把同一帧的输入、服务响应和最终显示结果集中管理，
    便于后续按帧序推进处理，避免服务响应先后顺序影响业务时序。
    """

    sequence_index: int
    request_id: str
    timestamp: Optional[float]
    image: np.ndarray
    created_at: float = field(default_factory=time.monotonic)
    upload_resp: Any = None
    upload_key: Optional[str] = None
    detection_resp: Any = None
    action_resp: Any = None
    action_submitted: bool = False
    detection_processed: bool = False
    output_ready: bool = False
    skipped: bool = False
    pixmap: Optional[QPixmap] = None
    mold_transition_state: ObjectState = ObjectState.DISAPPEARED
    mold_status: ResultState = ResultState.PENDING


class ActionClipBuffer:
    """Frame-ordered cache for one action-recognition clip.

    Upload keys are cached in frame order while material remains present, then
    submitted to VideoClassificationClient once the material explicitly
    disappears. Image saving and service requests may be consumed on different
    threads, so frame images are copied immediately when appended.

    动作识别片段缓存。
    物料持续出现期间按帧序缓存上传后的图片 key；物料明确消失时一次性提交给
    VideoClassificationClient。保存图片与服务请求可能在不同线程消费，因此缓存
    本地图片时立即复制，保证片段内容不受后续帧复用影响。
    """

    def __init__(self, max_frames: int):
        self.max_frames = max(1, max_frames)
        self.upload_keys: deque[str] = deque()
        self.frame_images: deque[tuple[int, np.ndarray]] = deque()

    def clear(self):
        self.upload_keys.clear()
        self.frame_images.clear()

    def append(self, sequence_index: int, upload_key: str, image: np.ndarray):
        """Append one uploaded frame to the clip in frame order.

        sequence_index 用于保持片段帧序；upload_key 是服务端缓存图片的标识；
        image 是当前帧的 RGB 图像数据。
        """
        if len(self.upload_keys) >= self.max_frames:
            self.upload_keys.popleft()
            self.frame_images.popleft()
        self.upload_keys.append(upload_key)
        self.frame_images.append((sequence_index, copy_stable_frame(image)))

    def upload_keys_in_order(self) -> list[str]:
        return list(self.upload_keys)

    def frame_images_in_order(self) -> list[tuple[int, np.ndarray]]:
        return list(self.frame_images)

    def __len__(self):
        return len(self.upload_keys)


def save_segments(
    images: list[tuple[int, np.ndarray]],
    material_area_points: list[tuple[int, int]],
    root_dir: str,
):
    """Save an action clip to disk in sequence_index order.

    按 sequence_index 顺序保存动作片段图片；文件名使用零填充序号，确保文件系统
    排序与视频帧序一致。material_area_points 为物料区域点列表，裁剪时使用
    前两个点；少于两个点时保存完整画面；root_dir 为保存根目录。
    """
    dirname = secrets.token_hex(6)
    full_dirname = osp.join(root_dir, dirname)
    os.makedirs(full_dirname, exist_ok=True)

    h, w = images[0][1].shape[:2] if images else (0, 0)
    xmin, xmax, ymin, ymax = 0, w, 0, h
    if len(material_area_points) >= 2:
        xmin = min(material_area_points[0][0], material_area_points[1][0])
        xmax = max(material_area_points[0][0], material_area_points[1][0])
        ymin = min(material_area_points[0][1], material_area_points[1][1])
        ymax = max(material_area_points[0][1], material_area_points[1][1])

    for sequence_index, image in images:
        # Zero-padding keeps filesystem order identical to frame order.
        image_path = osp.join(full_dirname, f"{sequence_index:08d}.jpg")
        image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(image_path, image_bgr[ymin:ymax, xmin:xmax])

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

        try:
            save_segments(data["images"], data["roi"], root)
        except Exception:
            logger.exception("保存分割图片失败")

    logger.info("退出保存结果线程")


class InnerScreenMicroscopicExaminationClient(QObject):
    """内屏镜检动作检测。内部保证异步响应按帧序处理。"""

    resultsReady = Signal(dict)
    imageReady = Signal(QPixmap)

    def __init__(self, settings: QSettings, parent=None):
        super().__init__(parent)
        self._settings = settings
        self.client_id = get_machine_unique_id()
        self.results = OrderedDict()
        self.threads: list[tuple[QThread, ClientBase]] = []
        self.image_queue = queue.Queue(1000)
        self._save_segments_threading: Optional[threading.Thread] = None

        self._sequence_index = -1
        self._records_by_request_id: dict[str, FrameRecord] = {}
        self._records_by_sequence: dict[int, FrameRecord] = {}
        self._next_detection_sequence = 0
        self._next_output_sequence = 0

        self._material_tracker = StateTracker()
        self._mold_tracker = StateTracker()
        self._clip_buffer = ActionClipBuffer(self._max_clip_frames())
        self._action_result_queue: deque[ResultState] = deque()
        self._current_action = ResultState.PENDING
        self._mold_status = ResultState.PENDING
        self._ok_count = 0
        self._total_count = 0

        self._drain_timer = QTimer(self)
        self._drain_timer.setInterval(200)
        self._drain_timer.timeout.connect(self._on_drain_timer)

    def _request_timeout_seconds(self) -> float:
        timeout_msecs = self._settings.value("request_timeout_msecs", 3000, type=int)
        return timeout_msecs / 1000.0

    def _max_clip_frames(self) -> int:
        return self._settings.value("max_clip_frames", 24, type=int)

    def clear_image_queue(self):
        self._reset_runtime_state()

    def _reset_runtime_state(self):
        self.results = OrderedDict()
        self._records_by_request_id.clear()
        self._records_by_sequence.clear()
        self._next_detection_sequence = self._sequence_index + 1
        self._next_output_sequence = self._sequence_index + 1
        self._material_tracker.reset()
        self._mold_tracker.reset()
        self._clip_buffer = ActionClipBuffer(self._max_clip_frames())
        self._action_result_queue.clear()
        self._current_action = ResultState.PENDING
        self._mold_status = ResultState.PENDING
        self._ok_count = 0
        self._total_count = 0

    def init_client(self):
        try:
            self.upload_image_client = UploadImageClient(self.client_id, IMAGE_MODEL_NAME)
            self.mold_detection_client = DetectionClient(self.client_id, MOLD_DETECTION_MODEL_NAME)
            self.action_client = VideoClassificationClient(self.client_id, ACTION_MODEL_NAME)

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
        if self.threads:
            logger.warning(f"{self.__class__.__name__} 已启动，忽略重复启动")
            return
        logger.info(f"启动 {self.__class__.__name__}")
        self.image_queue = queue.Queue(1000)
        self._clip_buffer = ActionClipBuffer(self._max_clip_frames())
        self._start_client_thread(self.upload_image_client)
        self._start_client_thread(self.mold_detection_client)
        self._start_client_thread(self.action_client)
        self._save_segments_threading = threading.Thread(
            target=backend_save_segments,
            args=(self.image_queue, osp.join(current_dir, "history")),
        )
        self._save_segments_threading.start()
        self._drain_timer.start()

    def _start_client_thread(self, client_obj: ClientBase):
        thread = QThread()
        client_obj.moveToThread(thread)
        thread.started.connect(client_obj.predict_unary)
        thread.start()
        self.threads.append((thread, client_obj))

    def stop(self):
        logger.info(f"停止 {self.__class__.__name__}")
        self._drain_timer.stop()
        for thread, client_obj in self.threads:
            client_obj.stop()
        for thread, _ in self.threads:
            thread.quit()
            thread.wait()
        self.threads.clear()
        if self._save_segments_threading and self._save_segments_threading.is_alive():
            self.image_queue.put(None)
            self._save_segments_threading.join(timeout=5)
            if self._save_segments_threading.is_alive():
                logger.warning("保存结果线程未能在超时时间内退出")

    def handle_image(self, image, image_encode=None, request_id=None, timestamp=None):
        if request_id is None:
            request_id = secrets.token_hex(4)

        # 入口处立即获得独立帧副本：后续上传、检测、显示都以该记录为准，
        # 不再直接引用宿主程序可能复用的帧缓冲区。
        owned_image = copy_stable_frame(image)
        self._sequence_index += 1
        record = FrameRecord(
            sequence_index=self._sequence_index,
            request_id=request_id,
            timestamp=timestamp,
            image=owned_image,
        )
        self._records_by_request_id[request_id] = record
        self._records_by_sequence[record.sequence_index] = record
        self.results.setdefault(request_id, {})["image"] = owned_image

        # 上传线程和本地记录生命周期不同，上传前再复制一份，避免客户端内部异步
        # 读取时与本地绘制/保存共享同一块 numpy 内存。
        upload_image = copy_stable_frame(owned_image)
        self.upload_image_client.add_input_item(
            {"request_id": request_id, "image": upload_image, "image_encode": image_encode}
        )
        return request_id

    @Slot(object, object)
    def _on_image_result(self, request_id, resp):
        record = self._records_by_request_id.get(request_id)
        if record is None or record.skipped:
            logger.warning(f"收到未知或已跳过的上传结果: {request_id}")
            return

        record.upload_resp = resp
        self.results.setdefault(request_id, {})["upload"] = resp
        if not getattr(resp, "cache_metas", None):
            logger.warning(f"上传结果缺少缓存 key: {request_id}")
            return

        record.upload_key = resp.cache_metas[0].key
        self.mold_detection_client.add_input_item(
            {"request_id": request_id, "image": None, "key": [record.upload_key]}
        )
        self._drain_detection_in_order()

    @Slot(object, object)
    def on_mold_detection(self, request_id, resp):
        record = self._records_by_request_id.get(request_id)
        if record is None or record.skipped:
            logger.warning(f"收到未知或已跳过的检测结果: {request_id}")
            return

        record.detection_resp = resp
        self.results.setdefault(request_id, {})["detection"] = resp
        self._drain_detection_in_order()

    @Slot(object, object)
    def on_action_recognition(self, request_id, resp):
        record = self._records_by_request_id.get(request_id)
        if record is None or record.skipped:
            logger.warning(f"收到未知或已跳过的动作识别结果: {request_id}")
            return

        record.action_resp = resp
        if record.detection_processed:
            record.output_ready = True
        self._drain_ready_outputs()

    def _on_drain_timer(self):
        self._drain_detection_in_order()
        self._drain_ready_outputs()

    def _is_expired(self, record: FrameRecord) -> bool:
        return time.monotonic() - record.created_at >= self._request_timeout_seconds()

    def _drain_detection_in_order(self):
        """Process detection results strictly by sequence_index.

        gRPC responses may arrive out of order, so this method only consumes
        the frame pointed to by _next_detection_sequence. If that frame has no
        detection result before the configured timeout, it is skipped so later
        completed frames are not blocked forever.

        按 sequence_index 顺序处理检测结果。
        gRPC 响应可能乱序返回，因此这里只消费 _next_detection_sequence 指向的帧。
        如果该帧迟迟没有检测结果，则在配置的超时时间后跳过它，避免后续
        已完成的帧被永久阻塞。
        """
        while True:
            record = self._records_by_sequence.get(self._next_detection_sequence)
            if record is None:
                break

            if record.detection_resp is None:
                if self._is_expired(record):
                    self._skip_record(record, "等待上传/检测结果超时")
                    self._next_detection_sequence += 1
                    continue
                break

            self._process_detection_record(record)
            self._next_detection_sequence += 1

        self._drain_ready_outputs()

    def _skip_record(self, record: FrameRecord, reason: str):
        """Drop an unusable frame and remove all indexes for late responses.

        All indexes are cleaned synchronously so an old asynchronous response
        cannot be matched again or emitted after this frame has been skipped.

        跳过无法继续处理的帧，并同步清理所有索引，防止旧响应再次参与输出。
        """
        record.skipped = True
        logger.warning(
            f"跳过帧 seq={record.sequence_index}, request_id={record.request_id}: {reason}"
        )
        self._records_by_request_id.pop(record.request_id, None)
        self._records_by_sequence.pop(record.sequence_index, None)
        self.results.pop(record.request_id, None)
        if record.sequence_index == self._next_output_sequence:
            self._next_output_sequence += 1

    def _process_detection_record(self, record: FrameRecord):
        if record.detection_processed:
            return

        # 物料状态只依据有明确匹配的物料框推进；未匹配到物料框时保持原状态，
        # 与重构前逻辑一致，避免因为某一帧漏检而过早提交动作识别请求。
        material_area = self._material_area(record.image)
        material_state = self._update_material_state(record, material_area)
        if material_state == ObjectState.APPEARED and record.upload_key:
            self._clip_buffer.append(record.sequence_index, record.upload_key, record.image)
        elif material_state == ObjectState.DISAPPEARING:
            self._submit_action_clip(record, material_area)
            self._material_tracker.reset()

        # 模具检测结果与动作识别结果最终都挂在同一 FrameRecord 上，后续统一按
        # sequence_index 输出，保证 UI 显示和统计口径与视频时间线一致。
        color, mold_state = self._update_mold_state(record)
        record.mold_transition_state = mold_state
        record.mold_status = self._mold_status
        record.pixmap = self.draw_detection_on_image(record.image, self._first_detection_result(record), color)
        record.detection_processed = True
        record.output_ready = not record.action_submitted

    def _update_material_state(self, record: FrameRecord, material_area) -> ObjectState:
        material_box = self._box_from_area(material_area)
        result = self._first_detection_result(record)
        if result is None:
            return self._material_tracker.state

        for box in result.boxes:
            if box.id not in [MATERIAL_EMPTY_BOX_ID, MATERIAL_PRESENT_BOX_ID]:
                continue
            if compute_iou([box.x_min, box.y_min, box.x_max, box.y_max], material_box) < 0.3:
                continue
            if box.id == MATERIAL_PRESENT_BOX_ID:
                return self._material_tracker.appear()
            if box.id == MATERIAL_EMPTY_BOX_ID:
                return self._material_tracker.disappear()

        return self._material_tracker.state

    def _submit_action_clip(self, record: FrameRecord, material_area):
        if not self._clip_buffer:
            self._clip_buffer.clear()
            return

        # VideoClassificationClient 要求 key 顺序与视频片段帧序一致；这里直接使用
        # ActionClipBuffer 中按 sequence_index 追加的顺序，不再受异步返回顺序影响。
        upload_keys = self._clip_buffer.upload_keys_in_order()
        request = {
            "request_id": record.request_id,
            "sequences_keys": [upload_keys],
            "sequences_rois": [[material_area for _ in upload_keys]],
        }
        self.action_client.add_input_item(request)
        record.action_submitted = True

        if self._settings.value("save_clip", False, type=bool):
            try:
                self.image_queue.put_nowait(
                    {
                        "images": self._clip_buffer.frame_images_in_order(),
                        "roi": material_area,
                    }
                )
            except queue.Full:
                logger.exception("保存图片失败")

        self._clip_buffer.clear()

    def _update_mold_state(self, record: FrameRecord) -> tuple[QColor, ObjectState]:
        mold_area = self._mold_area(record.image)
        result = self._first_detection_result(record)
        face_a = None
        face_c = None

        if result is not None:
            for box in result.boxes:
                center = ((box.x_min + box.x_max) / 2, (box.y_min + box.y_max) / 2)
                if mold_area and not in_polygon(center, mold_area):
                    continue
                if box.id == 0:
                    face_a = box
                elif box.id == 2:
                    face_c = box

        if not face_a or not face_c:
            color = QColor(0, 0, 255)
            if not face_a and not face_c:
                state = self._mold_tracker.disappear()
            else:
                state = self._mold_tracker.appear()
        elif face_a.y_min < face_c.y_min:
            color = QColor(255, 0, 0)
            state = self._mold_tracker.appear()
            self._mold_status = ResultState.NG
        else:
            color = QColor(0, 255, 0)
            state = self._mold_tracker.appear()
            self._mold_status = ResultState.OK

        if state == ObjectState.DISAPPEARING:
            self._mold_tracker.reset()
            self._mold_status = ResultState.PENDING

        return color, state

    def _drain_ready_outputs(self):
        """Emit completed frame results in video-frame order.

        The output stage also advances only the frame pointed to by
        _next_output_sequence, keeping UI refreshes, statistics, and resultsReady
        emissions aligned with the video timeline. If action recognition was
        submitted but exceeds the configured timeout, the frame is emitted as
        PENDING to avoid blocking screen refresh.

        按帧序输出已经准备好的结果。
        输出阶段同样只推进 _next_output_sequence 指向的帧，确保 UI、统计和外部
        resultsReady 信号都按视频时间线发布。若动作识别已提交但超过配置时间仍
        未返回，则按 PENDING 输出，避免单次动作服务异常阻塞整个画面刷新。
        """
        while True:
            record = self._records_by_sequence.get(self._next_output_sequence)
            if record is None:
                break

            if record.action_submitted and not record.output_ready and self._is_expired(record):
                logger.warning(
                    f"动作识别超时，按 PENDING 输出 seq={record.sequence_index}, request_id={record.request_id}"
                )
                record.output_ready = True

            if not record.output_ready:
                break

            self._emit_record(record)
            self._records_by_sequence.pop(record.sequence_index, None)
            self._records_by_request_id.pop(record.request_id, None)
            self.results.pop(record.request_id, None)
            self._next_output_sequence += 1

    def _emit_record(self, record: FrameRecord):
        action_resp = record.action_resp
        if action_resp is not None and getattr(action_resp, "results", None):
            label = action_resp.results[0].label
            if label in ["OK", "NG"]:
                self._current_action = ResultState.OK if label == "OK" else ResultState.NG
                self._action_result_queue.append(self._current_action)
                logger.debug(f"当前动作：{label}")

        # 动作识别结果在物料片段结束后才有业务意义；模具从出现到消失时统计一次。
        # 如果动作结果先于模具结束返回，暂存在队列中，等模具 DISAPPEARING 时消费。
        action_result = self._current_action
        if record.mold_transition_state == ObjectState.DISAPPEARING:
            action_result = self._dequeue_action_result()
            self._total_count += 1
            if action_result == ResultState.OK:
                self._ok_count += 1
            elif action_result == ResultState.PENDING:
                logger.warning("未获取到动作结果，当前动作结果None")
            logger.info(
                f"当前已做{self._total_count}, 一次通过率为{100 * self._ok_count / self._total_count:.2f}"
            )
            self._current_action = ResultState.PENDING
        elif self._action_result_queue:
            action_result = self._action_result_queue[0]

        data = {
            "image": record.image,
            "upload": record.upload_resp,
            "detection": record.detection_resp,
            "action": record.action_resp,
            "drawn": record.pixmap,
            "result": {
                "total": self._total_count,
                "ok": self._ok_count,
                "current_action": action_result,
                "current_mold": record.mold_status,
            },
        }
        self.resultsReady.emit({"request_id": record.request_id, "resp": data})

        # record.pixmap 已包含检测框；这里仅叠加动作识别文字，保持绘制职责清晰。
        pixmap = record.pixmap or self.draw_detection_on_image(record.image, None, QColor(0, 0, 255))
        color = self._action_color(self._current_action)
        text = f"内屏镜检撕膜：{self._action_text(self._current_action)}, 亮度:{get_v_channel_brightness(record.image):.1f}"
        self.imageReady.emit(self.draw_action_on_pixmap(pixmap, (10, 50), text, color))

    def _dequeue_action_result(self) -> ResultState:
        if self._action_result_queue:
            return self._action_result_queue.popleft()
        return ResultState.PENDING

    def _first_detection_result(self, record: FrameRecord):
        if record.detection_resp is None or not getattr(record.detection_resp, "results", None):
            return None
        return record.detection_resp.results[0]

    def _mold_area(self, image: np.ndarray):
        return self._scaled_area(self._settings.value("mold/points", [], type=list), image)

    def _material_area(self, image: np.ndarray):
        return self._scaled_area(self._settings.value("material/points", [], type=list), image)

    def _scaled_area(self, points_value, image: np.ndarray):
        h, w, _ = image.shape
        area = []
        if points_value:
            for point in points_value[0]:
                area.append((int(point[0] * w), int(point[1] * h)))
        return area

    def _box_from_area(self, area):
        if len(area) < 2:
            return None
        return [
            min(area[0][0], area[1][0]),
            min(area[0][1], area[1][1]),
            max(area[0][0], area[1][0]),
            max(area[0][1], area[1][1]),
        ]

    def _action_text(self, state: ResultState) -> str:
        if state == ResultState.OK:
            return "OK"
        if state == ResultState.NG:
            return "NG"
        return "PENDING"

    def _action_color(self, state: ResultState) -> QColor:
        if state == ResultState.OK:
            return QColor(0, 255, 0)
        if state == ResultState.NG:
            return QColor(255, 0, 0)
        return QColor(0, 0, 255)

    def draw_action_on_pixmap(self, pixmap, coord, text, color):
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont("Arial", 32))
        painter.setPen(QPen(color, 3))
        painter.drawText(QPoint(coord[0], coord[1]), text)
        painter.end()
        return pixmap

    def draw_detection_on_image(self, frame_rgb, result, color):
        owned_frame_rgb = copy_stable_frame(frame_rgb)
        h, w, ch = owned_frame_rgb.shape
        # QImage must own its bytes because drawing happens after this local buffer is gone.
        qimg = QImage(owned_frame_rgb.tobytes(), w, h, ch * w, QImage.Format_RGB888).copy()
        pixmap = QPixmap.fromImage(qimg)

        if result is None:
            return pixmap

        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setFont(QFont("Arial", 32))
        pen_det = QPen(color, 3)
        names = result.names

        for box in result.boxes:
            x1, y1, x2, y2 = int(box.x_min), int(box.y_min), int(box.x_max), int(box.y_max)
            painter.setPen(pen_det)
            painter.drawRect(x1, y1, x2 - x1, y2 - y1)
            painter.drawText(QPoint(x1, y1 - 5), f"{names[box.id]} {box.confidence:.2f}")

        painter.end()
        return pixmap
