# -*- coding: utf-8 -*-
import sys
import os
import time
import json
import random
import shutil
import cv2
import numpy as np
import torch

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QPainter, QColor, QPolygon, QImage, QPixmap
from PyQt5.QtWidgets import QLabel, QFileDialog, QMessageBox, QListWidgetItem

from ultralytics import YOLO

from ui.detect_ui import Ui_MainWindow
from utils.line_draw import (
    load_poly_area_data_simple,
    load_region,
    save_region,
    draw_poly_area_dangerous,
    draw_region,
    person_in_poly_area_dangerous,
    plot_one_box2,
    region_from_points,
    required_point_count,
    region_to_polygon_points,
    SHAPE_POLYGON, SHAPE_RECTANGLE, SHAPE_TRIANGLE, SHAPE_CIRCLE,
)
from utils.detector import DetectionWorker
from utils.excel_report import write_report


o_n_x_scale = 1.0
o_n_y_scale = 1.0
ruqin_check_for_draw = False

DISPLAY_W = 720
DISPLAY_H = 540


def pick_device(preference):
    p = preference.lower()
    if p == 'cpu':
        return 'cpu', False
    if not torch.cuda.is_available():
        return 'cpu', False
    try:
        cc = torch.cuda.get_device_capability(0)
        can_fp16 = cc[0] >= 7
    except Exception:
        can_fp16 = False
    return '0', bool(can_fp16)


class MyLabel(QLabel):
    polygon_changed = pyqtSignal()
    draw_finished = pyqtSignal()  # 当一个形状刚画完（非多边形自动发；多边形需外部停止）

    def __init__(self, parent=None):
        super(MyLabel, self).__init__(parent)
        # 画布坐标下的点（用于 paintEvent 绘制）
        self._canvas_pts = []
        # 原图坐标下的点（用于区域对象、判定、保存 JSON）
        self._image_pts = []

        self.flag = False
        self.shape = SHAPE_POLYGON  # 默认多边形
        self.setStyleSheet("background-color: rgb(30, 30, 30); border-radius: 6px;")
        self.setAlignment(Qt.AlignCenter)
        self.setFixedSize(DISPLAY_W, DISPLAY_H)

    # ---- 向后兼容的属性别名：旧代码仍然读写 Polygon_origin2canvas_list ----
    @property
    def Polygon_origin2canvas_list(self):
        return self._image_pts

    @Polygon_origin2canvas_list.setter
    def Polygon_origin2canvas_list(self, value):
        self._image_pts = [list(p) for p in value] if value else []
        # 反向同步到画布坐标（用于加载 JSON 后能在画布上看到）
        if self._image_pts:
            self._canvas_pts = [
                [p[0] / max(o_n_x_scale, 1e-6), p[1] / max(o_n_y_scale, 1e-6)]
                for p in self._image_pts
            ]
        else:
            self._canvas_pts = []
        self.update()

    # ------------------------------------------------------------------
    def delete(self):
        self.clear()

    def setFlag(self, flag):
        self.flag = flag

    def setShape(self, shape):
        """切换绘制形状；会清空当前已有的点。"""
        if shape not in (SHAPE_POLYGON, SHAPE_RECTANGLE, SHAPE_TRIANGLE, SHAPE_CIRCLE):
            return
        self.shape = shape
        self.clear_polygon()

    def clear_polygon(self):
        self._canvas_pts = []
        self._image_pts = []
        self.update()
        self.polygon_changed.emit()

    def get_region(self):
        """返回原图坐标系下的 region dict；点数不够返回 None。"""
        return region_from_points(self.shape, self._image_pts)

    def _save_region_to_json(self):
        region = self.get_region()
        if region is None:
            return
        try:
            os.makedirs('ruqin', exist_ok=True)
            save_region('ruqin/ruqin.json', region)
        except Exception as e:
            print("[region] 保存 JSON 失败:", e, flush=True)

    def mouseMoveEvent(self, event):
        self.setMouseTracking(True)

    def mousePressEvent(self, event):
        if event.button() == Qt.RightButton:
            self.clear_polygon()
            return

        if not self.flag:
            return

        pt_pos = event.pos()
        cx, cy = pt_pos.x(), pt_pos.y()
        scaled_x = o_n_x_scale * cx
        scaled_y = o_n_y_scale * cy

        need = required_point_count(self.shape)  # None = polygon（不限）

        # 非多边形且已收集够点数：此次点击视为"重新开始一个形状"
        if need is not None and len(self._image_pts) >= need:
            self._canvas_pts = []
            self._image_pts = []

        self._canvas_pts.append([cx, cy])
        self._image_pts.append([scaled_x, scaled_y])

        # 检查是否刚好完成
        finished = False
        if need is not None and len(self._image_pts) >= need:
            finished = True

        # 落盘（点数够的时候才写）
        self._save_region_to_json()

        self.update()
        self.polygon_changed.emit()
        if finished:
            self.draw_finished.emit()

    def _draw_preview_shape(self, painter):
        """按当前 shape 在画布上画预览图形。"""
        pts = self._canvas_pts
        if not pts:
            return

        if self.shape == SHAPE_CIRCLE:
            if len(pts) >= 2:
                cx, cy = pts[0]
                ex, ey = pts[1]
                r = int(round(((ex - cx) ** 2 + (ey - cy) ** 2) ** 0.5))
                painter.drawEllipse(QtCore.QPoint(int(cx), int(cy)), r, r)
            return

        if self.shape == SHAPE_RECTANGLE:
            if len(pts) >= 2:
                x1, y1 = pts[0]
                x2, y2 = pts[1]
                rect = QtCore.QRect(
                    int(min(x1, x2)), int(min(y1, y2)),
                    int(abs(x2 - x1)), int(abs(y2 - y1))
                )
                painter.drawRect(rect)
            return

        if self.shape == SHAPE_TRIANGLE:
            if len(pts) >= 3:
                qpts = [QtCore.QPoint(int(p[0]), int(p[1])) for p in pts[:3]]
                painter.drawPolygon(QPolygon(qpts))
            elif len(pts) == 2:
                painter.drawLine(int(pts[0][0]), int(pts[0][1]),
                                 int(pts[1][0]), int(pts[1][1]))
            return

        # polygon
        if len(pts) >= 2:
            qpts = [QtCore.QPoint(int(p[0]), int(p[1])) for p in pts]
            painter.drawPolygon(QPolygon(qpts))

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        pen = QtGui.QPen(QColor(255, 50, 50), 3)
        painter.setPen(pen)
        brush = QtGui.QBrush(QColor(255, 50, 50, 60))
        painter.setBrush(brush)

        if not ruqin_check_for_draw and len(self._canvas_pts) >= 1:
            self._draw_preview_shape(painter)

        # 点标记（黄色小点）
        painter.setPen(QtGui.QPen(QColor(255, 255, 0), 5))
        for p in self._canvas_pts:
            painter.drawPoint(int(p[0]), int(p[1]))


class UI_Logic_Window(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super(UI_Logic_Window, self).__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)

        self.ui.checkBox.setChecked(False)

        self.num_stop = 1
        self.output_folder = 'output/'

        self.draw_area = 0
        self.ruqin_check = False
        self.openfile_area = ''
        self.openfile_name_model = None
        self.model = None
        self.names = {}
        self.colors = []
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.imgsz = 640
        self.device = 'cpu'
        self.half = False

        self.draw_label = None
        self.Draw = ""
        self.worker = None

        # 通过 JSON 导入的 region（绘制的 region 从 draw_label 取）
        self.imported_region = None

        self.last_output_path = None
        self.last_output_kind = None

        self.allowed_classes = None

        self.pending_source = None
        self.pending_source_kind = None
        self.pending_image_data = None

        self.image_log = []
        self.image_session_info = None

        os.makedirs(self.output_folder + 'img_output', exist_ok=True)
        os.makedirs(self.output_folder + 'video_output', exist_ok=True)
        os.makedirs('ruqin', exist_ok=True)
        os.makedirs('weights', exist_ok=True)

        self.init_slots()
        self._update_device_label()
        self._reset_stats_display()

    def init_slots(self):
        self.ui.pushButton_2.clicked.connect(self.botton_area_open)
        self.ui.checkBox.stateChanged.connect(self.ruqin_flag)
        self.ui.pushButton_img.clicked.connect(self.select_image)
        self.ui.pushButton_video.clicked.connect(self.select_video)
        self.ui.pushButton_camer.clicked.connect(self.select_camera)
        self.ui.pushButton_start.clicked.connect(self.start_detection)
        self.ui.pushButton_weights.clicked.connect(self.open_model)
        self.ui.pushButton_init.clicked.connect(self.model_init)
        self.ui.pushButton_stop.clicked.connect(self.button_video_stop)
        self.ui.pushButton_finish.clicked.connect(self.finish_detect)
        self.ui.pushButton.clicked.connect(self.DrawPolygon)
        self.ui.pushButton_download.clicked.connect(self.download_result)

        self.ui.listWidget_classes.itemChanged.connect(self._on_class_filter_changed)
        self.ui.pushButton_select_all.clicked.connect(self._select_all_classes)
        self.ui.pushButton_select_none.clicked.connect(self._select_none_classes)
        self.ui.pushButton_invert.clicked.connect(self._invert_class_selection)
        self.ui.pushButton_yaml.clicked.connect(self.import_yaml_classes)

        self.ui.comboBox_shape.currentIndexChanged.connect(self._on_shape_changed)

    def _update_device_label(self):
        if torch.cuda.is_available():
            name = torch.cuda.get_device_name(0)
            self.setWindowTitle("YOLO 区域入侵检测系统 [GPU: {}]".format(name))
        else:
            self.setWindowTitle("YOLO 区域入侵检测系统 [CPU]")

    def _reset_stats_display(self):
        self.ui.lbl_current_value.setText("0")
        self.ui.lbl_nonintrusion_value.setText("0")
        self.ui.lbl_totalframes_value.setText("0")

    def _update_stats_display(self, stats):
        self.ui.lbl_current_value.setText(str(stats.get('current', 0)))
        self.ui.lbl_nonintrusion_value.setText(str(stats.get('nonintrusion', 0)))
        self.ui.lbl_totalframes_value.setText(str(stats.get('total_frames', 0)))

    def _populate_class_list(self):
        self.ui.listWidget_classes.blockSignals(True)
        self.ui.listWidget_classes.clear()

        if isinstance(self.names, dict):
            items = sorted(self.names.items(), key=lambda x: int(x[0]))
        else:
            items = [(i, n) for i, n in enumerate(self.names)]

        for cls_id, cls_name in items:
            item = QListWidgetItem("{}: {}".format(cls_id, cls_name))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked)
            item.setData(Qt.UserRole, int(cls_id))
            self.ui.listWidget_classes.addItem(item)

        self.ui.listWidget_classes.blockSignals(False)
        self._sync_allowed_classes()

    def _sync_allowed_classes(self):
        allowed = set()
        total = self.ui.listWidget_classes.count()
        for i in range(total):
            item = self.ui.listWidget_classes.item(i)
            if item.checkState() == Qt.Checked:
                allowed.add(int(item.data(Qt.UserRole)))

        if total == 0 or len(allowed) == total:
            self.allowed_classes = None
        else:
            self.allowed_classes = allowed

        if self.worker is not None and self.worker.isRunning():
            self.worker.update_allowed_classes(self.allowed_classes)

    def _on_class_filter_changed(self, item):
        self._sync_allowed_classes()

    def _select_all_classes(self):
        self.ui.listWidget_classes.blockSignals(True)
        for i in range(self.ui.listWidget_classes.count()):
            self.ui.listWidget_classes.item(i).setCheckState(Qt.Checked)
        self.ui.listWidget_classes.blockSignals(False)
        self._sync_allowed_classes()

    def _select_none_classes(self):
        self.ui.listWidget_classes.blockSignals(True)
        for i in range(self.ui.listWidget_classes.count()):
            self.ui.listWidget_classes.item(i).setCheckState(Qt.Unchecked)
        self.ui.listWidget_classes.blockSignals(False)
        self._sync_allowed_classes()

    def _invert_class_selection(self):
        self.ui.listWidget_classes.blockSignals(True)
        for i in range(self.ui.listWidget_classes.count()):
            item = self.ui.listWidget_classes.item(i)
            item.setCheckState(Qt.Unchecked if item.checkState() == Qt.Checked else Qt.Checked)
        self.ui.listWidget_classes.blockSignals(False)
        self._sync_allowed_classes()

    def import_yaml_classes(self):
        try:
            import yaml
        except ImportError:
            QMessageBox.warning(self, "提示", "未安装 pyyaml，请先 pip install pyyaml")
            return

        path, _ = QFileDialog.getOpenFileName(
            self, "选择数据集YAML文件", "", "YAML (*.yaml *.yml);;All Files (*)"
        )
        if not path:
            return

        try:
            with open(path, 'r', encoding='utf-8') as f:
                data = yaml.safe_load(f)
        except Exception as e:
            QMessageBox.warning(self, "警告", "YAML解析失败: " + str(e))
            return

        if not isinstance(data, dict) or 'names' not in data:
            QMessageBox.warning(self, "警告", "YAML中未找到 'names' 字段")
            return

        names_field = data['names']
        new_names = {}
        if isinstance(names_field, list):
            for i, n in enumerate(names_field):
                new_names[i] = str(n)
        elif isinstance(names_field, dict):
            for k, v in names_field.items():
                try:
                    new_names[int(k)] = str(v)
                except Exception:
                    pass
        else:
            QMessageBox.warning(self, "警告", "YAML中 'names' 格式不受支持")
            return

        if not new_names:
            QMessageBox.warning(self, "警告", "YAML中类别为空")
            return

        self.names = new_names
        name_count = len(new_names)
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(max(name_count, 80))]
        self._populate_class_list()

        QMessageBox.information(
            self, "已导入",
            "已从 {} 导入 {} 个类别\n\n提示：类别ID需要与模型输出一致才能正确筛选".format(
                os.path.basename(path), name_count
            )
        )

    def ruqin_flag(self):
        global ruqin_check_for_draw
        self.ruqin_check = self.ui.checkBox.isChecked()
        ruqin_check_for_draw = self.ruqin_check
        self._push_region_to_worker()

    def botton_area_open(self):
        self.openfile_area, _ = QFileDialog.getOpenFileName(
            self, '上传区域入侵json文件', 'ruqin/', '*.json'
        )
        if not self.openfile_area:
            return

        try:
            region = load_region(self.openfile_area)
        except Exception as e:
            QMessageBox.warning(self, "警告", "JSON解析失败: " + str(e))
            return

        self.draw_area = 1
        self.imported_region = region

        if self.draw_label is None:
            self._ensure_draw_label(False)

        # 同步到 draw_label：形状 + 原图点
        shape = region.get('shape', SHAPE_POLYGON)
        self.draw_label.shape = shape
        self._sync_shape_combo(shape)

        if shape == SHAPE_CIRCLE:
            # 圆形：以 [圆心, 圆心右侧 radius 像素] 作为两点
            cx, cy = region['center']
            r = region['radius']
            self.draw_label.Polygon_origin2canvas_list = [[cx, cy], [cx + r, cy]]
        else:
            self.draw_label.Polygon_origin2canvas_list = [list(p) for p in region['points']]

        self._push_region_to_worker()

    def _sync_shape_combo(self, shape):
        """把 UI 下拉框同步到给定 shape（不触发信号）。"""
        self.ui.comboBox_shape.blockSignals(True)
        for i in range(self.ui.comboBox_shape.count()):
            if self.ui.comboBox_shape.itemData(i) == shape:
                self.ui.comboBox_shape.setCurrentIndex(i)
                break
        self.ui.comboBox_shape.blockSignals(False)

    def _on_shape_changed(self, index):
        """用户切换了形状：清空当前绘制的区域，恢复"绘制区域"按钮。"""
        shape = self.ui.comboBox_shape.itemData(index)
        if shape is None:
            return
        self.imported_region = None
        self.draw_area = 0
        self.openfile_area = ''
        if self.draw_label is not None:
            self.draw_label.setShape(shape)
            self.draw_label.setFlag(False)
        self.ui.pushButton.setText("绘制区域")
        self._push_region_to_worker()

    def open_model(self):
        path, _ = QFileDialog.getOpenFileName(
            self, '选择模型权重文件', 'weights/', 'PyTorch (*.pt);;All Files (*)'
        )
        if not path:
            return
        self.openfile_name_model = path
        print('[model] 已选择权重：', path, flush=True)

    def get_model_path(self):
        if self.openfile_name_model:
            return self.openfile_name_model
        select_value = self.ui.comboBox.currentText()
        if select_value == '请选择模型':
            return 'yolov8n.pt'
        local = os.path.join('weights', select_value + '.pt')
        if os.path.exists(local):
            return local
        return select_value + '.pt'

    def model_init(self):
        model_path = self.get_model_path()
        device_pref = self.ui.comboBox_device.currentText()
        self.device, self.half = pick_device(device_pref)
        print("[init] 模型路径:", model_path, "| 设备:", self.device, "| half:", self.half, flush=True)

        try:
            print("[init] 正在加载权重 ...", flush=True)
            t0 = time.time()
            self.model = YOLO(model_path)
            print("[init] 权重加载完成，用时 {:.2f}s".format(time.time() - t0), flush=True)

            try:
                self.model.to(self.device)
                print("[init] 模型已移至设备:", self.device, flush=True)
            except Exception as e:
                print("[init] model.to 失败:", e, flush=True)

            self.names = self.model.names
            name_count = len(self.names) if self.names else 80
            self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(max(name_count, 80))]

            print("[init] 开始 warmup（首次 CUDA 初始化可能耗时 5-30s）...", flush=True)
            t1 = time.time()
            dummy = np.zeros((self.imgsz, self.imgsz, 3), dtype=np.uint8)
            try:
                _ = self.model.predict(
                    source=dummy, conf=self.conf_thres, iou=self.iou_thres,
                    imgsz=self.imgsz, device=self.device, half=self.half, verbose=False
                )
                print("[init] warmup 完成，用时 {:.2f}s".format(time.time() - t1), flush=True)
            except Exception as e:
                print("[init] warmup 失败，关闭 half 重试:", e, flush=True)
                self.half = False
                _ = self.model.predict(
                    source=dummy, conf=self.conf_thres, iou=self.iou_thres,
                    imgsz=self.imgsz, device=self.device, half=False, verbose=False
                )
                print("[init] FP32 warmup 完成，用时 {:.2f}s".format(time.time() - t1), flush=True)

            self._populate_class_list()

            if self.device != 'cpu':
                dev_name = "GPU ({})".format(torch.cuda.get_device_name(0))
            else:
                dev_name = "CPU"
            mode = "FP16" if self.half else "FP32"
            model_file = os.path.basename(model_path)
            QMessageBox.information(
                self, "消息",
                "模型加载完成\n权重: {}\n类别数: {}\n设备: {} ({})".format(
                    model_file, name_count, dev_name, mode
                )
            )
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "警告", "模型加载失败: " + str(e))
            self.model = None

    def _clear_center_layout(self):
        while self.ui.verticalLayout_5.count() > 0:
            item = self.ui.verticalLayout_5.takeAt(0)
            w = item.widget()
            if w is not None:
                w.setParent(None)
                w.deleteLater()

    def _ensure_draw_label(self, draw_flag=False):
        self._clear_center_layout()
        self.draw_label = MyLabel(self)
        # 应用下拉框当前形状
        shape = self.ui.comboBox_shape.currentData() or SHAPE_POLYGON
        self.draw_label.shape = shape
        self.draw_label.setFlag(draw_flag)
        self.draw_label.polygon_changed.connect(self._push_region_to_worker)
        self.draw_label.draw_finished.connect(self._on_draw_finished)
        self.ui.verticalLayout_5.addWidget(self.draw_label, 0, Qt.AlignCenter)

    def _on_draw_finished(self):
        """非多边形形状（矩形/三角形/圆形）绘制完一个后自动关闭绘制模式。"""
        if self.draw_label is None:
            return
        if self.draw_label.shape != SHAPE_POLYGON:
            self.draw_label.setFlag(False)
            self.ui.pushButton.setText("绘制区域")

    def _get_active_region(self):
        """取得当前生效的 region：优先使用 draw_label 上的；否则用导入的 JSON。"""
        if self.draw_label is not None:
            r = self.draw_label.get_region()
            if r is not None:
                return r
        if self.imported_region is not None:
            return self.imported_region
        return None

    def _push_region_to_worker(self):
        if self.worker is None:
            return
        region = self._get_active_region()
        self.worker.update_region(region, self.ruqin_check)

    def _set_frame_to_label(self, frame):
        origin_size = frame.shape
        global o_n_x_scale, o_n_y_scale
        o_n_x_scale = origin_size[1] / DISPLAY_W
        o_n_y_scale = origin_size[0] / DISPLAY_H

        show = cv2.resize(frame, (DISPLAY_W, DISPLAY_H))
        show = cv2.cvtColor(show, cv2.COLOR_BGR2RGB)
        show = np.ascontiguousarray(show)
        qimg = QImage(show.data, show.shape[1], show.shape[0], show.strides[0], QImage.Format_RGB888)
        if self.draw_label is not None:
            self.draw_label.setPixmap(QPixmap.fromImage(qimg))
            self.draw_label.setCursor(Qt.CrossCursor)

    def _on_frame_ready(self, frame, info):
        if self.draw_label is None:
            return
        self._set_frame_to_label(frame)
        self.ui.textBrowser.setText(info)

    def _on_stats_updated(self, stats):
        self._update_stats_display(stats)

    def _on_worker_error(self, msg):
        QMessageBox.warning(self, "警告", msg)

    def _on_worker_finished(self):
        self.ui.pushButton_video.setDisabled(False)
        self.ui.pushButton_img.setDisabled(False)
        self.ui.pushButton_camer.setDisabled(False)
        self.ui.pushButton_start.setDisabled(False)

    def select_image(self):
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先初始化模型")
            return

        self._stop_worker_if_running()

        img_name, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "data/images", "*.jpg;;*.png;;*.jpeg;;All Files(*)"
        )
        if not img_name:
            return

        img = cv2.imread(img_name)
        if img is None:
            QMessageBox.warning(self, "警告", "无法读取图片")
            return

        self._ensure_draw_label(False)
        self._reset_stats_display()
        self.ui.textBrowser.setText(
            '已加载图片。\n'
            '可勾选"区域入侵"进行入侵监测，也可直接\n'
            '点击 ▶ 开始检测 进行普通检测（可筛选类别）'
        )

        self.pending_source = img_name
        self.pending_source_kind = 'image'
        self.pending_image_data = img.copy()

        self._set_frame_to_label(img)

    def select_video(self):
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先初始化模型")
            return

        video_name, _ = QFileDialog.getOpenFileName(
            self, "选择视频", "data/", "*.mp4;;*.avi;;*.mov;;All Files(*)"
        )
        if not video_name:
            return

        self._stop_worker_if_running()

        cap_preview = cv2.VideoCapture(video_name)
        ret, frame = cap_preview.read()
        cap_preview.release()
        if not ret or frame is None:
            QMessageBox.warning(self, "警告", "无法读取视频首帧")
            return

        self._ensure_draw_label(False)
        self._reset_stats_display()
        self.ui.textBrowser.setText(
            '已加载视频（首帧预览）。\n'
            '可勾选"区域入侵"进行入侵监测，也可直接\n'
            '点击 ▶ 开始检测 进行普通检测（可筛选类别）'
        )

        self.pending_source = video_name
        self.pending_source_kind = 'video'
        self.pending_image_data = None

        self._set_frame_to_label(frame)

    def define_video_stream(self):
        video_dict = {
            '请选择视频流': '0',
            '本地摄像头': '0',
            '视频流1': 'rtsp://example.com/stream1',
            '视频流2': 'rtsp://example.com/stream2',
        }
        select_value = self.ui.comboBox_2.currentText()
        return video_dict.get(select_value, '0')

    def select_camera(self):
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先初始化模型")
            return

        self._stop_worker_if_running()

        video_value = self.define_video_stream()
        source = 0 if video_value == '0' else video_value

        cap_preview = cv2.VideoCapture(source)
        if not cap_preview.isOpened():
            cap_preview.release()
            QMessageBox.warning(self, "警告", "无法打开摄像头/视频流")
            return
        ret, frame = cap_preview.read()
        cap_preview.release()
        if not ret or frame is None:
            QMessageBox.warning(self, "警告", "摄像头/视频流读取失败")
            return

        self._ensure_draw_label(False)
        self._reset_stats_display()
        self.ui.textBrowser.setText(
            '摄像头/视频流已就绪。\n'
            '可勾选"区域入侵"进行入侵监测，也可直接\n'
            '点击 ▶ 开始检测 进行普通检测（可筛选类别）'
        )

        self.pending_source = source
        self.pending_source_kind = 'camera'
        self.pending_image_data = None

        self._set_frame_to_label(frame)

    def _is_polygon_ready(self):
        return self._get_active_region() is not None

    def start_detection(self):
        if self.model is None:
            QMessageBox.warning(self, "警告", "请先初始化模型")
            return

        if self.pending_source_kind is None:
            QMessageBox.warning(self, "警告", "请先选择图片/视频/摄像头作为检测源")
            return

        if self.ruqin_check and not self._is_polygon_ready():
            QMessageBox.warning(self, "提示",
                                '已勾选"区域入侵"但未设置区域：\n'
                                '• 点击"绘制区域"在画面上绘制\n'
                                '  （多边形≥3点；矩形2点；三角形3点；圆形2点），或\n'
                                '• 点击"上传区域(JSON)"加载已有区域\n'
                                '\n或取消勾选"区域入侵"以使用普通检测模式')
            return

        self._reset_stats_display()
        self.num_stop = 1
        self.ui.pushButton_stop.setText("暂停/继续")

        if self.pending_source_kind == 'image':
            self._run_image_detection()
        elif self.pending_source_kind == 'video':
            save_path = self._build_save_path('video')
            self.last_output_path = save_path
            self.last_output_kind = 'video'
            self._start_worker(self.pending_source, save_path)
        elif self.pending_source_kind == 'camera':
            save_path = self._build_save_path('video')
            self.last_output_path = save_path
            self.last_output_kind = 'video'
            self._start_worker(self.pending_source, save_path)

    def _run_image_detection(self):
        img = self.pending_image_data.copy()
        t0 = time.time()
        info_show, detections, pt_count, is_intrusion_active = self._detect_image(img)
        t1 = time.time()

        now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        ext = str(self.pending_source).split('.')[-1] if self.pending_source else 'jpg'
        file_path = os.path.join(self.output_folder, 'img_output', now + '.' + ext)
        cv2.imwrite(file_path, img)

        self.last_output_path = file_path
        self.last_output_kind = 'image'

        self.image_log = []
        ts = time.strftime("%H:%M:%S", time.localtime(time.time()))
        for d in detections:
            self.image_log.append({
                'frame': 1, 'time': ts,
                'class_id': d['cls'], 'class_name': d['name'],
                'confidence': round(d['conf'], 4),
                'x1': d['x1'], 'y1': d['y1'], 'x2': d['x2'], 'y2': d['y2'],
                'in_area': d['in_area'],
            })

        current_count = pt_count if is_intrusion_active else 0
        nonintrusion_count = len(detections) if not is_intrusion_active else 0
        self._update_stats_display({
            'current': current_count,
            'nonintrusion': nonintrusion_count,
            'total_frames': 1,
        })

        self.image_session_info = {
            'source_kind': 'image',
            'source': self.pending_source,
            'start_time_str': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0)),
            'end_time_str': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t1)),
            'duration_sec': round(t1 - t0, 3),
            'total_frames': 1,
            'intrusion_detection_count': pt_count if is_intrusion_active else 0,
            'total_detections': len(detections),
            'output_path': file_path,
            'ruqin': is_intrusion_active,
        }

        self.ui.textBrowser.setText(info_show)
        self._set_frame_to_label(img)

    def _detect_image(self, img):
        showimg = img
        detect_info_lines = []
        pt_count = 0
        detections = []
        active_region = None

        # ---- 1) 只计算 active_region，绝对不要把区域先画到 showimg 上 ----
        # 之前的实现把半透明红色区域画在 showimg 上之后再送去推理，模型会把
        # 这个红色形状当成目标产生幻觉（尤其是三角形/圆形在纯色背景里）。
        if self.ruqin_check:
            active_region = self._get_active_region()

        # ---- 2) 用干净原图做推理 ----
        try:
            results = self.model.predict(
                source=showimg, conf=self.conf_thres, iou=self.iou_thres,
                imgsz=self.imgsz, device=self.device, half=self.half, verbose=False
            )
        except Exception as e:
            if self.half:
                self.half = False
                results = self.model.predict(
                    source=showimg, conf=self.conf_thres, iou=self.iou_thres,
                    imgsz=self.imgsz, device=self.device, half=False, verbose=False
                )
            else:
                raise

        # ---- 3) 推理完成后再把区域画到显示帧上（在 bbox 之下） ----
        if active_region is not None:
            draw_region(showimg, active_region)

        if len(results) > 0 and results[0].boxes is not None and len(results[0].boxes) > 0:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            confs = results[0].boxes.conf.cpu().numpy()
            clss = results[0].boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                xyxy = boxes[i]
                conf = float(confs[i])
                cls = int(clss[i])

                if self.allowed_classes is not None and cls not in self.allowed_classes:
                    continue

                cls_name = self.names[cls] if isinstance(self.names, dict) else self.names[cls]
                label = '%s %.2f' % (cls_name, conf)
                color = self.colors[cls % len(self.colors)]
                x1, y1, x2, y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])

                if active_region is not None:
                    # person_in_poly_area_dangerous 已兼容 region dict
                    if person_in_poly_area_dangerous(xyxy, active_region):
                        s = plot_one_box2(xyxy, showimg, color=color, label=label, line_thickness=2)
                        pt_count += 1
                        detect_info_lines.append(s)
                        detections.append({
                            'cls': cls, 'name': cls_name, 'conf': conf,
                            'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'in_area': True,
                        })
                else:
                    s = plot_one_box2(xyxy, showimg, color=color, label=label, line_thickness=2)
                    detect_info_lines.append(s)
                    detections.append({
                        'cls': cls, 'name': cls_name, 'conf': conf,
                        'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2, 'in_area': False,
                    })

        is_intrusion_active = active_region is not None
        filter_desc = ""
        if self.allowed_classes is not None:
            names_list = []
            for c in sorted(self.allowed_classes):
                names_list.append(self.names[c] if isinstance(self.names, dict) else self.names[c])
            filter_desc = "筛选类别: {}\n".format(", ".join(names_list) if names_list else "(无)")
        if is_intrusion_active:
            shape_desc_map = {
                SHAPE_POLYGON: "多边形", SHAPE_RECTANGLE: "矩形",
                SHAPE_TRIANGLE: "三角形", SHAPE_CIRCLE: "圆形",
            }
            sd = shape_desc_map.get(active_region.get('shape', ''), '')
            header = ("区域入侵监测已开启（{}）\n".format(sd) + filter_desc
                      + "当前非法入侵物体数：{}\n".format(pt_count) + "-" * 30 + "\n")
        else:
            header = "普通检测模式\n" + filter_desc + "共检测到 {} 个目标\n".format(len(detect_info_lines)) + "-" * 30 + "\n"

        return header + "\n".join(detect_info_lines), detections, pt_count, is_intrusion_active

    def _build_save_path(self, kind):
        now = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime(time.time()))
        if kind == 'image':
            return os.path.join(self.output_folder, 'img_output', now + '.jpg')
        return os.path.join(self.output_folder, 'video_output', now + '.mp4')

    def _start_worker(self, source, save_path):
        self.worker = DetectionWorker(
            model=self.model, names=self.names, colors=self.colors,
            source=source, save_path=save_path,
            conf_thres=self.conf_thres, iou_thres=self.iou_thres,
            imgsz=self.imgsz, device=self.device, half=self.half,
        )
        self.worker.frame_ready.connect(self._on_frame_ready, Qt.QueuedConnection)
        self.worker.stats_updated.connect(self._on_stats_updated, Qt.QueuedConnection)
        self.worker.error_signal.connect(self._on_worker_error, Qt.QueuedConnection)
        self.worker.finished_signal.connect(self._on_worker_finished, Qt.QueuedConnection)
        self._push_region_to_worker()
        self.worker.update_allowed_classes(self.allowed_classes)

        self.ui.pushButton_video.setDisabled(True)
        self.ui.pushButton_img.setDisabled(True)
        self.ui.pushButton_camer.setDisabled(True)
        self.ui.pushButton_start.setDisabled(True)

        self.worker.start()

    def button_video_stop(self):
        if self.worker is None or not self.worker.isRunning():
            return

        if self.num_stop % 2 == 1:
            self.ui.pushButton_stop.setText("继续检测")
            self.worker.set_pause(True)
        else:
            self.ui.pushButton_stop.setText("暂停检测")
            self.worker.set_pause(False)
        self.num_stop += 1

    def _stop_worker_if_running(self):
        if self.worker is not None and self.worker.isRunning():
            self.worker.stop()
            self.worker.wait(3000)
            if self.worker.isRunning():
                self.worker.terminate()
                self.worker.wait(1000)

    def finish_detect(self):
        self._stop_worker_if_running()

        self.ui.pushButton_video.setDisabled(False)
        self.ui.pushButton_img.setDisabled(False)
        self.ui.pushButton_camer.setDisabled(False)
        self.ui.pushButton_start.setDisabled(False)
        self.ui.textBrowser.clear()

        if self.draw_label is not None:
            self.draw_label.clear_polygon()
        self._clear_center_layout()
        self.draw_label = None

        self.imported_region = None
        self.draw_area = 0
        self.openfile_area = ''

        self.pending_source = None
        self.pending_source_kind = None
        self.pending_image_data = None

        self.ui.pushButton.setText("绘制区域")
        self.ui.pushButton_stop.setText("暂停/继续")
        self.num_stop = 1

    def _build_session_info(self):
        if self.pending_source_kind == 'image' and self.image_session_info is not None:
            info = dict(self.image_session_info)
        elif self.worker is not None:
            total_detections = len(self.worker.frame_log)
            intrusion_count = sum(1 for e in self.worker.frame_log if e.get('in_area'))
            start = self.worker.session_start or time.time()
            end = self.worker.session_end or time.time()
            info = {
                'source_kind': self.pending_source_kind,
                'source': self.pending_source,
                'start_time_str': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(start)),
                'end_time_str': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end)),
                'duration_sec': round(end - start, 2),
                'total_frames': self.worker.stats.get('total_frames', 0),
                'intrusion_detection_count': intrusion_count,
                'total_detections': total_detections,
                'output_path': self.last_output_path or '',
                'ruqin': self.ruqin_check,
            }
        else:
            info = {}

        info['model_path'] = self.get_model_path()
        info['class_count'] = len(self.names) if self.names else 0
        if self.allowed_classes is None:
            info['allowed_classes_desc'] = '全部'
        else:
            names_list = []
            for c in sorted(self.allowed_classes):
                if isinstance(self.names, dict):
                    names_list.append(self.names.get(c, str(c)))
                elif c < len(self.names):
                    names_list.append(self.names[c])
            info['allowed_classes_desc'] = ", ".join(names_list) if names_list else "(空)"
        info['device_desc'] = ("GPU ({})".format(torch.cuda.get_device_name(0))
                               if self.device != 'cpu' and torch.cuda.is_available()
                               else "CPU")
        info['half'] = self.half

        region = self._get_active_region()
        if region is None:
            info['polygon_points'] = 0
            info['region_shape'] = '（未设置）'
        else:
            shape = region.get('shape', SHAPE_POLYGON)
            shape_desc_map = {
                SHAPE_POLYGON: '多边形', SHAPE_RECTANGLE: '矩形',
                SHAPE_TRIANGLE: '三角形', SHAPE_CIRCLE: '圆形',
            }
            if shape == SHAPE_CIRCLE:
                info['polygon_points'] = '圆心+半径={}px'.format(region.get('radius', 0))
            else:
                info['polygon_points'] = len(region.get('points') or [])
            info['region_shape'] = shape_desc_map.get(shape, shape)

        return info

    def _collect_frame_log(self):
        if self.pending_source_kind == 'image':
            return list(self.image_log)
        if self.worker is not None:
            return list(self.worker.frame_log)
        return []

    def download_result(self):
        if not self.last_output_path or not os.path.exists(self.last_output_path):
            QMessageBox.warning(self, "提示", "暂无可下载的检测结果，请先完成一次检测")
            return

        if self.worker is not None and self.worker.isRunning():
            reply = QMessageBox.question(
                self, "提示",
                "检测正在进行中，视频文件尚未关闭，现在下载可能不完整。\n是否仍要下载？",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        directory = QFileDialog.getExistingDirectory(self, "选择保存文件夹")
        if not directory:
            return

        src = self.last_output_path
        media_name = os.path.basename(src)
        base_name = os.path.splitext(media_name)[0]
        media_dst = os.path.join(directory, media_name)
        excel_dst = os.path.join(directory, base_name + "_report.xlsx")

        saved = []
        try:
            shutil.copy2(src, media_dst)
            saved.append(media_dst)
        except Exception as e:
            QMessageBox.warning(self, "错误", "媒体文件保存失败: " + str(e))
            return

        try:
            session_info = self._build_session_info()
            frame_log = self._collect_frame_log()
            write_report(excel_dst, session_info, frame_log)
            saved.append(excel_dst)
        except Exception as e:
            import traceback
            traceback.print_exc()
            QMessageBox.warning(self, "警告", "Excel报表生成失败: " + str(e))

        QMessageBox.information(self, "完成", "已保存:\n" + "\n".join(saved))

    def DrawPolygon(self):
        if self.draw_label is None:
            return
        shape_label_map = {
            SHAPE_POLYGON: "多边形",
            SHAPE_RECTANGLE: "矩形",
            SHAPE_TRIANGLE: "三角形",
            SHAPE_CIRCLE: "圆形",
        }
        if self.ui.pushButton.text() == '绘制区域':
            # 开启前：同步下拉框选中的形状，并清空之前的点
            shape = self.ui.comboBox_shape.currentData() or SHAPE_POLYGON
            self.draw_label.setShape(shape)
            self.imported_region = None
            self.draw_area = 0
            self.openfile_area = ''

            self.Draw = shape
            self.draw_label.setFlag(True)
            self.ui.pushButton.setText('停止绘制')

            tip_map = {
                SHAPE_POLYGON: "多边形模式：依次单击>=3个点，再次点按钮结束；右键清空。",
                SHAPE_RECTANGLE: "矩形模式：单击 2 个对角点自动完成；右键清空。",
                SHAPE_TRIANGLE: "三角形模式：单击 3 个顶点自动完成；右键清空。",
                SHAPE_CIRCLE: "圆形模式：单击圆心 + 边缘 2 个点自动完成；右键清空。",
            }
            self.ui.textBrowser.setText(tip_map.get(shape, "开始绘制"))
        else:
            self.draw_label.setFlag(False)
            self.ui.pushButton.setText('绘制区域')

    def closeEvent(self, event):
        self._stop_worker_if_running()
        event.accept()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    current_ui = UI_Logic_Window()
    current_ui.setMouseTracking(True)
    current_ui.show()
    sys.exit(app.exec_())
