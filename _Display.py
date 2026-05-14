import os
import os.path as osp
import sys

import cv2
import numpy as np
from hummingbirdai.widgets import MultiDrawingCanvas, ShapeType
from PySide6.QtCore import (Property, QPoint, QPropertyAnimation, QRect,
                            QSettings, Qt, QTimer, Signal)
from PySide6.QtGui import (QAction, QBrush, QColor, QFont, QIcon, QImage,
                           QPainter, QPen, QPixmap)
from PySide6.QtWidgets import (QApplication, QCheckBox, QGridLayout, QGroupBox,
                               QHBoxLayout, QLabel, QMenu, QPushButton,
                               QSlider, QStyle, QStyleOptionSlider, QTextEdit,
                               QVBoxLayout, QWidget)

current_file_path = osp.abspath(__file__)  # 当前文件的绝对路径
current_dir = osp.dirname(current_file_path)  # 当前文件所在目录


class DisplayWidget(QWidget):
    """测试插件的显示widget"""

    polygonFinished: Signal = Signal(list)
    rectangleFinished: Signal = Signal(list)

    def __init__(self, settings: QSettings):
        super().__init__()
        self._settings = settings
        self.setup_ui()
        # 1) 启用自定义右键菜单策略
        self.setContextMenuPolicy(Qt.CustomContextMenu)

        # 2) 连接信号
        self.customContextMenuRequested.connect(self.show_context_menu)

    def show_context_menu(self, pos: QPoint):
        """
        在这里构建并弹出右键菜单
        :param pos: 相对于控件自身的坐标
        """
        menu = QMenu(self)

        edit_mold_region = QAction("绘制上下模区域", self)
        edit_mold_region.setIcon(QIcon(os.path.join(current_dir, "icons/paint.png")))
        edit_material_region = QAction("绘制物料区域", self)
        edit_material_region.setIcon(
            QIcon(os.path.join(current_dir, "icons/paint.png"))
        )
        remove_region = QAction("删除区域", self)
        remove_region.setIcon(QIcon(os.path.join(current_dir, "icons/delete.png")))

        # 连接动作
        edit_mold_region.triggered.connect(
            lambda: self.toggleDrawing(ShapeType.POLYGON)
        )

        edit_material_region.triggered.connect(
            lambda: self.toggleDrawing(ShapeType.RECTANGLE)
        )

        remove_region.triggered.connect(lambda: self.image_label.delete_shape_at(pos))

        menu.addAction(edit_mold_region)
        menu.addAction(edit_material_region)
        menu.addAction(remove_region)

        # 3) 在全局坐标下弹出菜单
        global_pos = self.mapToGlobal(pos)
        menu.exec(global_pos)

    def setup_ui(self):
        layout = QVBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)

        # 图像显示区域
        self.image_label = MultiDrawingCanvas()

        self.bottom_label = QLabel()

        show_grid = QGridLayout()
        show_grid.addWidget(self.bottom_label, 0, 0)
        show_grid.addWidget(self.image_label, 0, 0)

        self.image_label.polygonFinished.connect(
            lambda points, shape_type=ShapeType.POLYGON: self.save_shape(
                shape_type, points, "mold"
            )
        )

        self.image_label.rectangleFinished.connect(
            lambda points, shape_type=ShapeType.RECTANGLE: self.save_shape(
                shape_type, points, "material"
            )
        )

        self.load_shape("mold")
        self.load_shape("material")

        self.image_label.setScaledContents(True)
        self.bottom_label.setScaledContents(True)
        self.bottom_label.setMinimumSize(640, 480)
        self.bottom_label.setAlignment(Qt.AlignCenter)
        self.image_label.setStyleSheet("background-color: transparent; color: red;")
        self.bottom_label.setStyleSheet(
            "background-color: #2b2b2b; border: 1px solid #555; color: #ccc; font-size: 16px;"
        )
        self.bottom_label.setText("插件页")

        # 添加一个标识标签
        title_label = QLabel(f" 监销视频分析 ")
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet(
            "font-size: 20px; font-weight: bold; color: blue; background-color: yellow; padding: 5px;"
        )

        # layout.addWidget(title_label, stretch=0)
        # layout.addWidget(self.image_label, stretch=1)
        layout.addLayout(show_grid, stretch=1)
        self.setLayout(layout)

    def update_image(self, image):
        """更新显示的图像"""

        if not self.isVisible():
            return

        try:
            if isinstance(image, QImage):
                pixmap = QPixmap.fromImage(image)
            else:
                pixmap = image

            # scaled_pixmap = pixmap.scaled(
            #     self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            # )
            self.bottom_label.setPixmap(pixmap)

        except Exception as e:
            self.bottom_label.setText(f"图像显示错误: {str(e)}")

    def save_shape(self, shape_type, points_list, group=None):

        if group:
            self._settings.beginGroup(group)

        shape_type_value = shape_type.value
        self._settings.setValue("shape_type", shape_type_value)
        self._settings.setValue("points", points_list)
        if group:
            self._settings.endGroup()

    def load_shape(self, group=None):

        if group:
            self._settings.beginGroup(group)

        shape_type_value = self._settings.value("shape_type", type=int)
        if shape_type_value:
            shape_type = ShapeType(shape_type_value)
            points_list = self._settings.value("points", [], type=list)

            for points in points_list:
                self.image_label.loadShape(shape_type, points)
        if group:
            self._settings.endGroup()

    def toggleDrawing(self, shape_type: ShapeType) -> None:
        self.image_label.startDrawing(shape_type)

    def delete_shape(self, shape: ShapeType):
        self.image_label.delete_shape(shape)
