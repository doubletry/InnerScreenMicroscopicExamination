import os
import sys

import cv2
import grpc
import numpy as np
from PySide6.QtCore import (Property, QPropertyAnimation, QRect, QSettings, Qt,
                            QTimer, Signal)
from PySide6.QtGui import (QAction, QBrush, QColor, QFont, QImage, QPainter,
                           QPen, QPixmap)
from PySide6.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QLineEdit, QMessageBox,
                               QPushButton, QSlider, QStyle,
                               QStyleOptionSlider, QTextEdit, QVBoxLayout,
                               QWidget)

from hummingbirdai.widgets import GRPCPanel, SliderwithLabel, Switch


class SidebarStatusWidget(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)

        # 统计数据
        self.total_count = 0
        self.ok_count = 0
        self.ng_count = 0

        self._init_ui()
        self._update_stats_labels()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(10)

        # ===================== 上半部分：实时信息 =====================
        realtime_group = QWidget(self)
        realtime_layout = QVBoxLayout(realtime_group)
        realtime_layout.setContentsMargins(0, 0, 0, 0)
        # realtime_layout.setSpacing(12)

        # 上半部分标题
        title_label_top = QLabel("实时信息", self)
        title_font = QFont()
        title_font.setBold(True)
        title_label_top.setFont(title_font)
        realtime_layout.addWidget(title_label_top, stretch=0)

        # 撕膜动作
        self.strip_status_icon = QLabel(self)
        self.strip_status_text = QLabel(self)
        self.strip_title_label = QLabel(self)
        strip_widget = self._create_status_block(
            "撕膜动作",
            self.strip_status_icon,
            self.strip_title_label,
            self.strip_status_text,
        )
        realtime_layout.addWidget(strip_widget, stretch=1)

        # 上下模摆放
        self.place_status_icon = QLabel(self)
        self.place_status_text = QLabel(self)
        self.place_title_label = QLabel(self)
        place_widget = self._create_status_block(
            "上下模摆放",
            self.place_status_icon,
            self.place_title_label,
            self.place_status_text,
        )
        realtime_layout.addWidget(place_widget, stretch=1)

        # realtime_layout.addStretch()

        # ===================== 下半部分：统计信息 =====================
        stats_group = QWidget(self)
        stats_layout = QVBoxLayout(stats_group)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(8)

        title_label_bottom = QLabel("统计信息", self)
        title_label_bottom.setFont(title_font)
        stats_layout.addWidget(title_label_bottom)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(4)

        # 标签
        self.label_total = QLabel("总数：", self)
        self.label_ok = QLabel("OK 数：", self)
        self.label_ng = QLabel("NG 数：", self)
        self.label_ok_rate = QLabel("OK 率：", self)
        self.label_ng_rate = QLabel("NG 率：", self)

        # 数值
        self.value_total = QLabel("0", self)
        self.value_ok = QLabel("0", self)
        self.value_ng = QLabel("0", self)
        self.value_ok_rate = QLabel("0.00%", self)
        self.value_ng_rate = QLabel("0.00%", self)

        # ===== 行 0：总数（浅蓝色背景）=====
        total_row = QWidget(self)
        total_row_layout = QGridLayout(total_row)
        total_row_layout.setContentsMargins(4, 2, 4, 2)
        total_row_layout.setHorizontalSpacing(12)

        total_row_layout.addWidget(self.label_total, 0, 0, Qt.AlignLeft)
        total_row_layout.addWidget(self.value_total, 0, 1, Qt.AlignRight)

        # 浅蓝色背景
        total_row.setStyleSheet("background-color: #E4F1FF; border-radius: 3px;")
        grid.addWidget(total_row, 0, 0, 1, 2)  # 占用两列

        # ===== 行 1：OK 数（浅绿色背景）=====
        ok_count_row = QWidget(self)
        ok_count_row_layout = QGridLayout(ok_count_row)
        ok_count_row_layout.setContentsMargins(4, 2, 4, 2)
        ok_count_row_layout.setHorizontalSpacing(12)

        ok_count_row_layout.addWidget(self.label_ok, 0, 0, Qt.AlignLeft)
        ok_count_row_layout.addWidget(self.value_ok, 0, 1, Qt.AlignRight)

        # 浅绿色背景，可以按需要调整颜色
        ok_count_row.setStyleSheet("background-color: #E3F6E3; border-radius: 3px;")
        grid.addWidget(ok_count_row, 1, 0, 1, 2)  # 占用两列

        # ===== 行 2：NG 数（浅红色背景）=====
        ng_count_row = QWidget(self)
        ng_count_row_layout = QGridLayout(ng_count_row)
        ng_count_row_layout.setContentsMargins(4, 2, 4, 2)
        ng_count_row_layout.setHorizontalSpacing(12)

        ng_count_row_layout.addWidget(self.label_ng, 0, 0, Qt.AlignLeft)
        ng_count_row_layout.addWidget(self.value_ng, 0, 1, Qt.AlignRight)

        ng_count_row.setStyleSheet("background-color: #FDE4E4; border-radius: 3px;")
        grid.addWidget(ng_count_row, 2, 0, 1, 2)

        # ===== 行 3：OK 率（浅绿色背景）=====
        ok_rate_row = QWidget(self)
        ok_rate_row_layout = QGridLayout(ok_rate_row)
        ok_rate_row_layout.setContentsMargins(4, 2, 4, 2)
        ok_rate_row_layout.setHorizontalSpacing(12)

        ok_rate_row_layout.addWidget(self.label_ok_rate, 0, 0, Qt.AlignLeft)
        ok_rate_row_layout.addWidget(self.value_ok_rate, 0, 1, Qt.AlignRight)

        ok_rate_row.setStyleSheet("background-color: #E3F6E3; border-radius: 3px;")
        grid.addWidget(ok_rate_row, 3, 0, 1, 2)

        # ===== 行 4：NG 率（浅红色背景）=====
        ng_rate_row = QWidget(self)
        ng_rate_row_layout = QGridLayout(ng_rate_row)
        ng_rate_row_layout.setContentsMargins(4, 2, 4, 2)
        ng_rate_row_layout.setHorizontalSpacing(12)

        ng_rate_row_layout.addWidget(self.label_ng_rate, 0, 0, Qt.AlignLeft)
        ng_rate_row_layout.addWidget(self.value_ng_rate, 0, 1, Qt.AlignRight)

        ng_rate_row.setStyleSheet("background-color: #FDE4E4; border-radius: 3px;")
        grid.addWidget(ng_rate_row, 4, 0, 1, 2)

        stats_layout.addLayout(grid)
        stats_layout.addStretch()

        # ===== 主布局：通过 stretch 设置 8:2 的高度比例 =====
        # 上半部分 : 下半部分 = 8 : 2
        main_layout.addWidget(realtime_group, stretch=8)
        main_layout.addWidget(stats_group, stretch=2)

        # 初始状态设为 NG
        self.set_strip_status(False)
        self.set_place_status(False)

    def _create_status_block(
        self, title: str, icon_label: QLabel, title_label: QLabel, status_label: QLabel
    ) -> QWidget:
        """
        实时状态区域：
            [图标（OK/NG）]
            [大号文字：撕膜动作 / 上下模摆放]
            [小号文字：当前状态 OK / NG]
        标题文字在图标下面，居中且字号更大。
        """
        block_widget = QWidget(self)
        block_layout = QVBoxLayout(block_widget)
        block_layout.setContentsMargins(0, 0, 0, 0)
        block_layout.setSpacing(4)

        block_layout.addStretch()
        # 图标（在上）
        icon_label.setFixedSize(128, 128)
        icon_label.setScaledContents(True)
        block_layout.addWidget(icon_label, alignment=Qt.AlignHCenter)

        # 大号标题文字（撕膜动作 / 上下模摆放）
        title_font = QFont()
        title_font.setPointSize(11)  # 可以根据实际大小调整
        title_font.setBold(True)
        title_label.setFont(title_font)
        title_label.setText(title)
        title_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        block_layout.addWidget(title_label)

        # 小号状态文字（OK/NG）
        status_font = QFont()
        status_font.setPointSize(9)
        status_label.setFont(status_font)
        status_label.setText("NG")
        status_label.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        block_layout.addWidget(status_label)

        block_layout.addStretch()

        return block_widget

    # ===================== 状态更新接口 =====================

    def set_strip_status(self, is_ok: bool):
        """
        设置撕膜动作状态：True=OK，False=NG
        """
        self._set_status(
            is_ok=is_ok,
            icon_label=self.strip_status_icon,
            status_label=self.strip_status_text,
        )

    def set_place_status(self, is_ok: bool):
        """
        设置上下模摆放状态：True=OK，False=NG
        """
        self._set_status(
            is_ok=is_ok,
            icon_label=self.place_status_icon,
            status_label=self.place_status_text,
        )

    def _set_status(self, is_ok: bool, icon_label: QLabel, status_label: QLabel):
        """
        根据 OK/NG 设置图标和文字颜色。
        当前用纯色方块模拟图标，你可以替换为实际 PNG/SVG 图标。
        """
        status_text = "OK" if is_ok else "NG"
        status_label.setText(status_text)

        color = QColor("#4CAF50") if is_ok else QColor("#F44336")  # 绿/红
        palette = status_label.palette()
        palette.setColor(status_label.foregroundRole(), color)
        status_label.setPalette(palette)

        # 简单的纯色块图标，有实际图标时可以替换为 QPixmap("ok.png") 等
        pix = QPixmap(40, 40)
        pix.fill(color)
        icon_label.setPixmap(pix)

    # ===================== 统计信息接口（外部设置总数和OK数） =====================

    def set_total_and_ok(self, total: int, ok: int):
        """
        外部设置统计数据：
            total: 总数
            ok:    OK 数
        内部自动计算：
            NG 数 = total - ok
            OK 率、NG 率
        """
        if total < 0:
            total = 0
        if ok < 0:
            ok = 0
        if ok > total:
            ok = total

        self.total_count = total
        self.ok_count = ok
        self.ng_count = self.total_count - self.ok_count

        self._update_stats_labels()

    def reset_stats(self):
        """
        清零统计信息。
        """
        self.total_count = 0
        self.ok_count = 0
        self.ng_count = 0
        self._update_stats_labels()

    def _update_stats_labels(self):
        """
        根据 total_count / ok_count / ng_count
        计算 OK/NG 率并刷新界面显示。
        """
        self.value_total.setText(str(self.total_count))
        self.value_ok.setText(str(self.ok_count))
        self.value_ng.setText(str(self.ng_count))

        if self.total_count == 0:
            ok_rate = 0.0
            ng_rate = 0.0
        else:
            ok_rate = self.ok_count / self.total_count * 100
            ng_rate = self.ng_count / self.total_count * 100

        self.value_ok_rate.setText(f"{ok_rate:.2f}%")
        self.value_ng_rate.setText(f"{ng_rate:.2f}%")
