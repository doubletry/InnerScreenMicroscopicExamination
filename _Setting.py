import os
import os.path as osp
import sys

import cv2
import grpc
import numpy as np
from PySide6.QtCore import (Property, QPropertyAnimation, QRect, QSettings, Qt,
                            QTime, QTimer, Signal)
from PySide6.QtGui import (QAction, QBrush, QColor, QFont, QIcon, QImage,
                           QPainter, QPen, QPixmap)
from PySide6.QtWidgets import (QApplication, QCheckBox, QDoubleSpinBox,
                               QFormLayout, QFrame, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QSizePolicy, QSlider, QSpacerItem,
                               QSpinBox, QStyle, QStyleOptionSlider, QTextEdit,
                               QVBoxLayout, QWidget)

from hummingbirdai.logger_config import logger
from hummingbirdai.widgets import (GRPCPanel, Switch, SwitchGroup,
                                   TimeRangeDialog, TimeSpinBox)

from ._util import tcp_request, tcp_request_async

current_file_path = os.path.abspath(__file__)  # 当前文件的绝对路径
current_dir = os.path.dirname(current_file_path)  # 当前文件所在目录


class ConfigurationPanel(QWidget):
    """测试插件的侧边栏widget"""

    def __init__(self, settings: QSettings):
        super().__init__()
        self._settings = settings
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)

        server_configuration_group = QGroupBox("服务器配置")
        server_configuration_group_layout = QFormLayout(server_configuration_group)

        server_ip_input = QLineEdit()
        server_ip_input.setObjectName("server_ip")

        server_ip_input.setText(
            self.load_setting(server_ip_input.objectName(), "127.0.0.1", str)
        )
        server_ip_input.textChanged.connect(
            lambda value: self.save_setting(server_ip_input.objectName(), value)
        )
        server_configuration_group_layout.addRow("服务器IP：", server_ip_input)

        image_port_input = QSpinBox()
        image_port_input.setObjectName("image_port")
        image_port_input.setRange(1, 65535)
        image_port_input.setValue(
            (self.load_setting(image_port_input.objectName(), 50050, int))
        )
        image_port_input.valueChanged.connect(
            lambda value: self.save_setting(image_port_input.objectName(), value)
        )
        server_configuration_group_layout.addRow("图像服务端口：", image_port_input)

        detection_port_input = QSpinBox()
        detection_port_input.setObjectName("detection_port")
        detection_port_input.setRange(1, 65535)
        detection_port_input.setValue(
            (self.load_setting(detection_port_input.objectName(), 50051, int))
        )
        detection_port_input.valueChanged.connect(
            lambda value: self.save_setting(detection_port_input.objectName(), value)
        )
        server_configuration_group_layout.addRow(
            "上下模检测端口：", detection_port_input
        )

        action_port_input = QSpinBox()
        action_port_input.setObjectName("action_port")
        action_port_input.setRange(1, 65535)
        action_port_input.setValue(
            (self.load_setting(action_port_input.objectName(), 50059, int))
        )
        action_port_input.valueChanged.connect(
            lambda value: self.save_setting(action_port_input.objectName(), value)
        )
        server_configuration_group_layout.addRow("动作识别端口：", action_port_input)

        param_configuration_group = QGroupBox("参数配置")
        param_configuration_group_layout = QFormLayout(param_configuration_group)
        detection_conf = QDoubleSpinBox()
        detection_conf.setObjectName("detection_conf")
        detection_conf.setRange(0.1, 1)
        detection_conf.setSingleStep(0.05)
        detection_conf.setValue(
            (self.load_setting(detection_conf.objectName(), 0.5, float))
        )
        detection_conf.valueChanged.connect(
            lambda value: self.save_setting(detection_conf.objectName(), value)
        )

        param_configuration_group_layout.addRow("上下模检测阈值阈值：", detection_conf)

        save_clip_checkbox = QCheckBox()
        save_clip_checkbox.setObjectName("save_clip")
        save_clip_checkbox.setChecked(
            self.load_setting(save_clip_checkbox.objectName(), False, bool)
        )
        save_clip_checkbox.checkStateChanged.connect(
            lambda value: self.save_setting(
                save_clip_checkbox.objectName(), value == Qt.Checked
            )
        )

        param_configuration_group_layout.addRow("保存片段", save_clip_checkbox)

        layout.addWidget(server_configuration_group)
        layout.addWidget(param_configuration_group)

    def save_enabled(self, value: bool, group=None):
        if group:
            self.save_setting(f"{group}/enable", value)
        else:
            self.save_setting("enable", value)

    def load_setting(self, key, default, type=None, group=None):
        if group:
            return self._settings.value(f"{group}/{key}", default, type=type)
        else:
            return self._settings.value(key, default, type=type)

    def save_setting(self, key, value, group: str = ""):
        if group:
            self._settings.beginGroup(group)
            self._settings.setValue(key, value)
            self._settings.endGroup()
        else:
            self._settings.setValue(key, value)

    def save_settings(self, configs: dict, group: str = ""):
        if group:
            self._settings.beginGroup(group)
            for k, v in configs.items():
                self._settings.setValue(k, v)
            self._settings.endGroup()
        else:
            for k, v in configs.items():
                self._settings.setValue(k, v)
