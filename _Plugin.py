import os
import os.path as osp
from collections import deque

import numpy as np
from hummingbirdai.logger_config import logger
from hummingbirdai.plugins import PluginBase
from hummingbirdai.ui import logoer
from hummingbirdai.widgets import AlarmToast, EventListWidget
from PySide6.QtCore import (Property, QPropertyAnimation, QRect, QSettings, Qt,
                            QTimer, Signal, Slot)
from PySide6.QtGui import (QAction, QBrush, QColor, QFont, QFontMetrics,
                           QImage, QPainter, QPen, QPixmap)
from PySide6.QtSvg import QSvgRenderer
from PySide6.QtWidgets import (QApplication, QCheckBox, QDialog, QGroupBox,
                               QHBoxLayout, QLabel, QMessageBox, QPushButton,
                               QSlider, QStyle, QStyleOptionSlider, QTextEdit,
                               QVBoxLayout, QWidget)

from ._Display import DisplayWidget
from ._Setting import ConfigurationPanel
from ._Sidebar import SidebarStatusWidget
from ._util import get_package_name, get_v_channel_brightness
from ._version import (__version__, compatibility, department, description,
                       organization, year)
from .core import InnerScreenMicroscopicExaminationClient

current_file_path = os.path.abspath(__file__)  # 当前文件的绝对路径
current_dir = os.path.dirname(current_file_path)  # 当前文件所在目录


def create_pixmap(
    text: str,
    size: int = 96,
    font_family: str = "Microsoft YaHei",
    output_path: str = None,
) -> QPixmap:
    """
    根据输入 text 生成正方形 QPixmap。
    - text 中的 '\\n' 表示换行。
    - 自动根据内容计算合适的字体大小，使文字尽量填满 size x size 的区域。
    - 如果 output_path 不为空，则将生成的 pixmap 保存到该路径。
    """

    # 预处理文本行
    lines = text.split("\n")

    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    font = QFont(font_family)
    font.setBold(True)  # 图标文字一般用粗体更清晰
    painter.setFont(font)

    # 留一点边距，避免文字贴边
    margin_ratio = 0.95  # 越接近 1 越“满”
    best_point_size = 10
    max_point_size = 200

    # 试探合适字号
    for point_size in range(10, max_point_size):
        font.setPointSize(point_size)
        painter.setFont(font)
        fm = QFontMetrics(font)

        # 计算所有行中最长行的宽度，以及总高度
        max_width = 0
        line_height = fm.height()
        for line in lines:
            w = fm.horizontalAdvance(line)
            if w > max_width:
                max_width = w

        total_height = line_height * len(lines)

        # 超出目标区域（考虑 margin）时停止
        if max_width > size * margin_ratio or total_height > size * margin_ratio:
            break

        best_point_size = point_size

    # 使用找到的最佳字号重新绘制
    font.setPointSize(best_point_size)
    painter.setFont(font)
    painter.setPen(QColor(0, 0, 0))

    fm = QFontMetrics(font)
    line_height = fm.height()
    total_height = line_height * len(lines)

    # 计算起始 y，使所有行整体垂直居中
    start_y = (size - total_height) / 2 + fm.ascent()

    # 按行绘制，每行水平居中
    for i, line in enumerate(lines):
        line_width = fm.horizontalAdvance(line)
        x = (size - line_width) / 2
        y = start_y + i * line_height
        painter.drawText(int(x), int(y), line)

    painter.end()

    # 如果指定了输出路径，则保存
    if output_path and not osp.exists(output_path):
        pixmap.save(output_path)

    return pixmap


class FrameEventSmoother:
    """
    帧事件平滑器：
    - 将每次 record 调用视为一帧
    - 只统计最近 window_frames 帧内的事件
    - 若某事件在窗口内出现次数 >= threshold_frames，则在当前帧返回该事件
    """

    def __init__(self, window_frames: int, threshold_frames: int) -> None:
        """
        :param threshold_frames: 触发阈值（窗口内至少出现多少帧的该事件）
        :param window_frames: 时间窗大小（按帧计）
        """
        self.threshold_frames = threshold_frames
        self.window_frames = window_frames

        self.init_queue()

    def record(self, event_key: str) -> str | None:
        """
        记录当前帧发生的事件，并判断是否“触发”。

        :param event_key: 当前帧的事件（str）
        :return: 若该事件在最近 window_frames 帧中出现次数 >= threshold_frames，
                 则返回 event_key，否则返回 None
        """
        # 1. 将当前事件放入窗口
        self._window.append(event_key)
        self._counts[event_key] = self._counts.get(event_key, 0) + 1
        # 2. 若窗口长度超过 window_frames，则移除最旧的一帧
        if len(self._window) > self.window_frames:
            old_event = self._window.popleft()
            self._counts[old_event] -= 1
            if self._counts[old_event] == 0:
                del self._counts[old_event]

        # 3. 判断当前事件在窗口内的出现次数是否达到阈值
        if self._counts.get(event_key, 0) >= self.threshold_frames:
            return event_key
        return None

    def init_queue(self):
        # 最近 window_frames 帧的事件序列
        self._window: deque[str] = deque()
        # 当前窗口内各事件出现次数
        self._counts: dict[str, int] = {}

    def clear(self):
        self.init_queue()

    def set_threshold_frames(self, threshold_frames: int):
        self.threshold_frames = threshold_frames

    def set_window_frames(self, window_frames: int):
        self.window_frames = window_frames


class Plugin(PluginBase):
    def __init__(self):
        # 插件基础信息

        self.version = __version__
        self.description = description

        super().__init__(get_package_name(), self.version)
        # 插件控件
        self.sidebar_widget = None
        self.display_widget = None
        self.main_window = None

        # 插件状态
        self.is_active = False

        # 工具栏动作
        self.start_action = None
        self.control_panel_action = None
        self.client = InnerScreenMicroscopicExaminationClient(self.settings)
        self.result_list_widget = SidebarStatusWidget(self.main_window)
        self.request_id2relativetime_map = {}

        sampling_window = self.settings.value("sampling_window", 15, type=int)
        anomaly_count = self.settings.value("anomaly_count", 5, type=int)
        self.smoother = FrameEventSmoother(sampling_window, anomaly_count)

        # 上下模状态
        self._mold_ok_count = 0
        self._mold_ng_count = 0
        self._mold_currend_state = None

    def get_name(self):
        return self.name

    def get_version(self):
        return self.version

    def get_description(self):
        return self.description

    def get_menu_actions(self):
        """返回菜单动作列表"""
        action = QAction("显示测试消息", None)
        action.triggered.connect(self.show_version_message)
        return [action]

    def get_toolbar_actions(self):
        """返回工具栏动作列表"""
        actions = []

        # 启动/停止插件动作
        if not self.start_action:
            self.start_action = QAction("启动插件", None)

            self.start_action.setIcon(logoer.get_icon("start"))
            self.start_action.setCheckable(True)
            self.start_action.triggered.connect(self.on_start_action_triggered)

        actions.append(self.start_action)

        # 配置参数动作
        set_parameter_action = QAction("配置参数", None)
        set_parameter_action.setIcon(logoer.get_icon("setting"))
        set_parameter_action.triggered.connect(self.show_configuration_dialog)
        actions.append(set_parameter_action)

        test_action = QAction("插件版本", None)
        test_action.setIcon(logoer.get_icon("info"))
        test_action.triggered.connect(self.show_version_message)
        actions.append(test_action)

        return actions

    def show_configuration_dialog(self):
        dialog = QDialog(self.main_window)
        dialog_layout = QVBoxLayout(dialog)
        configuration_panel = ConfigurationPanel(settings=self.settings)
        dialog_layout.addWidget(configuration_panel)

        dialog.setWindowTitle("配置参数")
        dialog.exec()  # 会非阻塞地显示对话框

        sampling_window = self.settings.value("sampling_window", 15, type=int)
        anomaly_count = self.settings.value("anomaly_count", 5, type=int)
        self.smoother = FrameEventSmoother(sampling_window, anomaly_count)

    def get_sidebar_widget(self):
        """返回侧边栏widget"""

        return self.result_list_widget

    def get_display_widget(self):
        """返回显示widget"""
        try:
            # 检查现有widget是否仍然有效
            if self.display_widget:
                # 尝试访问widget来检查它是否已被删除
                _ = self.display_widget.objectName()
            else:
                # 创建新的widget
                self.display_widget = DisplayWidget(settings=self.settings)
            return self.display_widget
        except RuntimeError:
            # 如果widget已被删除，创建新的
            self.display_widget = DisplayWidget(settings=self.settings)
            return self.display_widget

    def on_start_action_triggered(self, checked):
        """处理启动动作触发"""

        if checked:

            if not self.client.init_client():
                self.show_configuration_dialog()
                return

            # 启动插件
            self.start_action.setText("停止插件")
            self.start_action.setIcon(logoer.get_icon("stop"))

            self.client.start()
            try:
                self.client.imageReady.connect(
                    self.display_widget.update_image, Qt.UniqueConnection
                )
            except TypeError:
                pass  # 已经连接过

            try:
                self.client.resultsReady.connect(
                    self.analysis_video_info, Qt.UniqueConnection
                )
            except TypeError:
                pass  # 已经连接

        else:
            # 停止插件
            self.start_action.setText("启动插件")
            self.start_action.setIcon(logoer.get_icon("start"))
            self.client.stop()

    def activate(self):
        """激活插件"""
        self.is_active = True

    def initialize(self, main_window):
        """插件初始化，传入主窗口引用"""
        self.main_window = main_window
        AlarmToast.bind_main_window(self.main_window)

    def update_start_action_state(self, is_active):
        """更新启动动作的状态"""
        if self.start_action:
            self.start_action.setChecked(is_active)
            if is_active:
                self.start_action.setText("停止插件")
                self.start_action.setIcon(logoer.get_icon("stop"))
            else:
                self.start_action.setText("启动插件")
                self.start_action.setIcon(logoer.get_icon("start"))

    def deactivate(self):
        """停用插件"""
        self.is_active = False

    def cleanup(self):
        """插件清理"""
        # 停止处理帧数据

        self.client.stop()
        self.is_active = False

        # 清理引用并明确设置为None
        try:
            if self.sidebar_widget:
                self.sidebar_widget.deleteLater()
        except RuntimeError:
            pass

        try:
            if self.display_widget:
                self.display_widget.deleteLater()
        except RuntimeError:
            pass

        self.sidebar_widget = None
        self.display_widget = None

    def show_version_message(self):
        """显示测试消息"""
        msg = QMessageBox(self.main_window)
        msg.setWindowTitle("关于插件")

        logo_path = osp.join(current_dir, "icons", "logo.ico")
        if not osp.exists(logo_path):
            create_pixmap("内屏\n镜检", output_path=logo_path)

        pixmap = create_pixmap("内屏\n镜检")

        msg.setIconPixmap(pixmap)

        # 主标题（加粗、大号字体、蓝色）
        plugin_name = get_package_name().rsplit(".")[-1]
        msg.setText(
            f"<b><font size='+2' color='#2c3e50'>{plugin_name} v{__version__}</font></b>"
        )

        # 附加信息（多行、部分加粗）
        msg.setInformativeText(
            f"版权所有 <b>© {year} {organization}</b><br>"
            f"适用于 <font color='#27ae60'>{compatibility}</font><br>"
            f"部门：<font color='#2980b9'>{department}</font><br>"
            "官网：<a href='https://www.honor.com'>https://www.honor.com</a>"
        )

        # 让超链接可点击
        msg.setTextFormat(Qt.RichText)
        msg.setTextInteractionFlags(Qt.TextBrowserInteraction)
        # msg.setOpenExternalLinks(True)

        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec()

    def on_display_activated(self):
        """当显示被激活时调用"""
        self.is_active = True

    def on_display_deactivated(self):
        """当显示被停用时调用"""
        self.is_active = False

    @Slot(np.ndarray, float)
    def on_frame_received(self, frame_data: np.ndarray, timestamp: float):
        if not self.start_action.isChecked():
            return

        if get_v_channel_brightness(frame_data) < self.settings.value(
            "brighten_conf", 64, type=int
        ):  # 如果画面过暗，可能是视频结束的黑屏，直接返回不处理
            return

        request_id = self.client.handle_image(frame_data)
        self.request_id2relativetime_map[request_id] = timestamp

    @Slot(bytes, list, list)
    def on_segment_received(self, segment_data, frames, timestamps):
        pass

    def on_video_started(self, meta: dict = None):
        self.video_meta = meta

        self.result_list_widget.reset_stats()
        self.request_id2relativetime_map.clear()
        self.client.clear_image_queue()

    def on_video_finished(self):

        logger.info("视频结束")

    def analysis_video_info(self, data: dict):
        request_id = data.get("request_id")
        response_data: dict = data.get("resp")

        total_result = response_data["result"]
        total_count = total_result["total"]
        ok_count = total_result["ok"]
        current_action = total_result["current_action"]
        current_mold = total_result["current_mold"]

        self.result_list_widget.set_place_status(current_mold)
        self.result_list_widget.set_strip_status(current_action)

        self.result_list_widget.set_total_and_ok(total_count, ok_count)
