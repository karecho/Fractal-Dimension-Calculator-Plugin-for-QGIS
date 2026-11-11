"""QGIS plugin for image-based box-counting of subway line maps."""

import csv
import math
import os

from qgis.PyQt.QtCore import Qt, QSize
from qgis.PyQt.QtGui import QAction, QColor, QImage, QPainter, QPen, QPixmap
from qgis.PyQt.QtWidgets import (
    QComboBox, QDialog, QFileDialog, QFormLayout, QGroupBox, QHBoxLayout,
    QLabel, QLineEdit, QMessageBox, QProgressBar, QPushButton, QSpinBox,
    QTabWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)
from qgis.core import (
    QgsMapSettings, QgsMapRendererParallelJob, QgsProject, QgsRectangle,
)


def qt_enum(owner, legacy_name, group_name, member_name):
    """Resolve Qt5's flat and Qt6's scoped enum spellings."""
    if hasattr(owner, legacy_name):
        return getattr(owner, legacy_name)
    return getattr(getattr(owner, group_name), member_name)


IMAGE_ARGB32 = qt_enum(QImage, "Format_ARGB32", "Format", "Format_ARGB32")
IMAGE_RGB32 = qt_enum(QImage, "Format_RGB32", "Format", "Format_RGB32")
IMAGE_GRAYSCALE8 = qt_enum(QImage, "Format_Grayscale8", "Format", "Format_Grayscale8")
ALIGN_CENTER = qt_enum(Qt, "AlignCenter", "AlignmentFlag", "AlignCenter")
KEEP_ASPECT = qt_enum(Qt, "KeepAspectRatio", "AspectRatioMode", "KeepAspectRatio")
ANTIALIASING = qt_enum(QPainter, "Antialiasing", "RenderHint", "Antialiasing")
DASH_LINE = qt_enum(Qt, "DashLine", "PenStyle", "DashLine")


def square_extent(extent):
    """Smallest axis-aligned square containing the layer's bounding box."""
    if extent.isEmpty() or extent.width() <= 0 or extent.height() <= 0:
        raise ValueError("当前图层的范围为空或宽高为零，无法构造外接正方形。")
    side = max(extent.width(), extent.height())
    center = extent.center()
    half = side / 2.0
    return QgsRectangle(center.x() - half, center.y() - half,
                        center.x() + half, center.y() + half)


def image_gray_bytes(image):
    """Return tightly packed 8-bit grayscale bytes, compositing alpha on white."""
    canvas = QImage(image.size(), IMAGE_ARGB32)
    canvas.fill(QColor("white"))
    painter = QPainter(canvas)
    painter.drawImage(0, 0, image)
    painter.end()
    gray = canvas.convertToFormat(IMAGE_GRAYSCALE8)
    pointer = gray.constBits()
    if hasattr(pointer, "setsize"):
        pointer.setsize(gray.bytesPerLine() * gray.height())
    data = bytes(pointer)
    stride = gray.bytesPerLine()
    width = gray.width()
    if stride == width:
        return width, gray.height(), data
    return width, gray.height(), b"".join(
        data[y * stride:y * stride + width] for y in range(gray.height())
    )


def gray_image(width, height, data):
    """Construct a QImage via the portable binary PGM format."""
    return QImage.fromData("P5\n{} {}\n255\n".format(width, height).encode("ascii") + data)


def otsu_threshold(data):
    histogram = [0] * 256
    for value in data:
        histogram[value] += 1
    total = len(data)
    total_sum = sum(value * count for value, count in enumerate(histogram))
    background_count = 0
    background_sum = 0
    best_variance = -1.0
    threshold = 127
    for value, count in enumerate(histogram):
        background_count += count
        if background_count == 0:
            continue
        foreground_count = total - background_count
        if foreground_count == 0:
            break
        background_sum += value * count
        background_mean = background_sum / background_count
        foreground_mean = (total_sum - background_sum) / foreground_count
        variance = background_count * foreground_count * (background_mean - foreground_mean) ** 2
        if variance > best_variance:
            best_variance = variance
            threshold = value
    return threshold


def box_count(mask, width, height, min_box=2, max_box=1024):
    """Count boxes containing foreground pixels; include partial edge boxes."""
    sizes = []
    counts = []
    size = min_box
    grid_width = (width + size - 1) // size
    grid_height = (height + size - 1) // size
    grid = bytearray(grid_width * grid_height)
    for gy in range(grid_height):
        y0, y1 = gy * size, min((gy + 1) * size, height)
        for gx in range(grid_width):
            x0, x1 = gx * size, min((gx + 1) * size, width)
            if any(any(mask[y * width + x0:y * width + x1]) for y in range(y0, y1)):
                grid[gy * grid_width + gx] = 1
    while size <= max_box:
        count = sum(grid)
        sizes.append(size)
        counts.append(count)
        next_width = (grid_width + 1) // 2
        next_height = (grid_height + 1) // 2
        if size * 2 > max_box:
            break
        next_grid = bytearray(next_width * next_height)
        for gy in range(next_height):
            for gx in range(next_width):
                occupied = False
                for py in range(gy * 2, min(gy * 2 + 2, grid_height)):
                    start = py * grid_width + gx * 2
                    if any(grid[start:min(start + 2, (py + 1) * grid_width)]):
                        occupied = True
                        break
                if occupied:
                    next_grid[gy * next_width + gx] = 1
        grid, grid_width, grid_height = next_grid, next_width, next_height
        size *= 2
    if len(sizes) < 2:
        raise ValueError("需要至少两个箱尺度才能拟合分形维度。")
    return sizes, counts


def fit_dimension(sizes, counts):
    points = [(math.log(size, 2), math.log(count, 2))
              for size, count in zip(sizes, counts) if count > 0]
    if len(points) < 2:
        raise ValueError("有效箱计数少于两个尺度，无法拟合分形维度。")
    mean_x = sum(x for x, _ in points) / len(points)
    mean_y = sum(y for _, y in points) / len(points)
    variance_x = sum((x - mean_x) ** 2 for x, _ in points)
    if variance_x == 0:
        raise ValueError("有效箱尺度不足，无法拟合分形维度。")
    slope = sum((x - mean_x) * (y - mean_y) for x, y in points) / variance_x
    intercept = mean_y - slope * mean_x
    residual = sum((y - (slope * x + intercept)) ** 2 for x, y in points)
    total = sum((y - mean_y) ** 2 for _, y in points)
    r_squared = 1.0 if total == 0 and residual == 0 else (0.0 if total == 0 else 1.0 - residual / total)
    return -slope, max(0.0, min(1.0, r_squared)), points, slope, intercept


class FitPlot(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.points = []
        self.slope = 0.0
        self.intercept = 0.0
        self.setMinimumHeight(250)

    def set_data(self, points, slope, intercept):
        self.points, self.slope, self.intercept = points, slope, intercept
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(ANTIALIASING)
        painter.fillRect(self.rect(), QColor("white"))
        area = self.rect().adjusted(55, 18, -20, -45)
        painter.setPen(QPen(QColor("#667085"), 1))
        painter.drawLine(area.bottomLeft(), area.bottomRight())
        painter.drawLine(area.bottomLeft(), area.topLeft())
        painter.drawText(4, 20, "log₂ N")
        painter.drawText(area.center().x() - 28, self.height() - 10, "log₂(box width)")
        if len(self.points) < 2:
            painter.drawText(area, ALIGN_CENTER, "运行计算后显示拟合图")
            return
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        xmin, xmax = min(xs), max(xs)
        ymin, ymax = min(ys), max(ys)
        if xmax == xmin:
            xmax = xmin + 1
        if ymax == ymin:
            ymax = ymin + 1
        def project(x, y):
            return (area.left() + (x - xmin) / (xmax - xmin) * area.width(),
                    area.bottom() - (y - ymin) / (ymax - ymin) * area.height())
        painter.setPen(QPen(QColor("#1565c0"), 2))
        for x, y in self.points:
            px, py = project(x, y)
            painter.drawEllipse(int(px - 3), int(py - 3), 6, 6)
        painter.setPen(QPen(QColor("#e65100"), 2, DASH_LINE))
        p1 = project(xmin, self.slope * xmin + self.intercept)
        p2 = project(xmax, self.slope * xmax + self.intercept)
        painter.drawLine(int(p1[0]), int(p1[1]), int(p2[0]), int(p2[1]))


class AnalysisDialog(QDialog):
    def __init__(self, iface, parent=None):
        super().__init__(parent)
        self.iface = iface
        self.setWindowTitle("地铁线路图分形维度分析")
        self.resize(1000, 720)
        root = QVBoxLayout(self)

        settings_box = QGroupBox("输入与参数")
        form = QFormLayout(settings_box)
        self.layer_combo = QComboBox()
        for layer in QgsProject.instance().mapLayers().values():
            if layer.isValid():
                self.layer_combo.addItem(layer.name(), layer.id())
        active = iface.activeLayer()
        if active:
            index = self.layer_combo.findData(active.id())
            if index >= 0:
                self.layer_combo.setCurrentIndex(index)
        self.output_edit = QLineEdit(os.path.join(os.path.expanduser("~"), "subway_fractal_output"))
        browse = QPushButton("浏览…")
        browse.clicked.connect(self.choose_output)
        output_row = QWidget()
        output_layout = QHBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_edit)
        output_layout.addWidget(browse)
        self.resolution = QSpinBox()
        self.resolution.setRange(256, 4096)
        self.resolution.setSingleStep(256)
        self.resolution.setValue(1024)
        self.threshold_mode = QComboBox()
        self.threshold_mode.addItems(["自动阈值（Otsu）", "手动阈值"])
        self.threshold_value = QSpinBox()
        self.threshold_value.setRange(0, 255)
        self.threshold_value.setValue(127)
        self.threshold_mode.currentIndexChanged.connect(
            lambda index: self.threshold_value.setEnabled(index == 1))
        self.threshold_value.setEnabled(False)
        threshold_row = QWidget()
        threshold_layout = QHBoxLayout(threshold_row)
        threshold_layout.setContentsMargins(0, 0, 0, 0)
        threshold_layout.addWidget(self.threshold_mode)
        threshold_layout.addWidget(QLabel("阈值"))
        threshold_layout.addWidget(self.threshold_value)
        form.addRow("分析图层", self.layer_combo)
        form.addRow("输出目录", output_row)
        form.addRow("正方形图像边长（像素）", self.resolution)
        form.addRow("二值化设置", threshold_row)
        root.addWidget(settings_box)

        action_row = QHBoxLayout()
        self.status = QLabel("二值图中白色像素作为线路前景；箱宽由 2 像素起逐次翻倍，最大 1024 像素。")
        self.status.setWordWrap(True)
        self.progress = QProgressBar()
        self.progress.setRange(0, 0)
        self.progress.hide()
        self.run_button = QPushButton("开始分析")
        self.run_button.clicked.connect(self.process)
        action_row.addWidget(self.status, 1)
        action_row.addWidget(self.progress)
        action_row.addWidget(self.run_button)
        root.addLayout(action_row)

        self.tabs = QTabWidget()
        self.preview_labels = {}
        for key, title in (("rgb", "初始 RGB 图"), ("gray", "灰度图"), ("binary", "二值图")):
            label = QLabel("运行分析后显示预览")
            label.setAlignment(ALIGN_CENTER)
            label.setMinimumSize(420, 380)
            label.setStyleSheet("background:#f2f4f7; color:#667085;")
            self.preview_labels[key] = label
            self.tabs.addTab(label, title)

        result_tab = QWidget()
        result_layout = QVBoxLayout(result_tab)
        self.result_label = QLabel("分形维度与 R² 将在计算完成后显示。")
        self.result_label.setStyleSheet("font-size:16px; font-weight:bold; padding:8px;")
        self.plot = FitPlot()
        self.table = QTableWidget(0, 2)
        self.table.setHorizontalHeaderLabels(["箱宽（像素）", "非空箱数"])
        self.table.horizontalHeader().setStretchLastSection(True)
        result_layout.addWidget(self.result_label)
        result_layout.addWidget(self.plot, 1)
        result_layout.addWidget(self.table)
        self.tabs.addTab(result_tab, "箱计数与拟合")
        root.addWidget(self.tabs, 1)

    def choose_output(self):
        path = QFileDialog.getExistingDirectory(self, "选择输出目录", self.output_edit.text())
        if path:
            self.output_edit.setText(path)

    def _set_preview(self, key, image):
        pixmap = QPixmap.fromImage(image)
        smooth = qt_enum(Qt, "SmoothTransformation", "TransformationMode", "SmoothTransformation")
        self.preview_labels[key].setPixmap(pixmap.scaled(
            self.preview_labels[key].size(), KEEP_ASPECT, smooth))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        for key, image in getattr(self, "images", {}).items():
            self._set_preview(key, image)

    def process(self):
        layer_id = self.layer_combo.currentData()
        layer = QgsProject.instance().mapLayer(layer_id) if layer_id else None
        if layer is None or not layer.isValid():
            QMessageBox.warning(self, "无法处理", "请选择一个有效的 QGIS 图层。")
            return
        outdir = self.output_edit.text().strip()
        if not outdir:
            QMessageBox.warning(self, "无法处理", "请指定输出目录。")
            return
        self.run_button.setEnabled(False)
        self.progress.show()
        self.status.setText("正在渲染图层并处理图像…")
        from qgis.PyQt.QtWidgets import QApplication
        QApplication.processEvents()
        try:
            os.makedirs(outdir, exist_ok=True)
            extent = square_extent(layer.extent())
            settings = QgsMapSettings()
            settings.setLayers([layer])
            settings.setDestinationCrs(layer.crs())
            settings.setTransformContext(QgsProject.instance().transformContext())
            settings.setExtent(extent)
            size = self.resolution.value()
            settings.setOutputSize(QSize(size, size))
            settings.setOutputDpi(300)
            settings.setBackgroundColor(QColor("white"))
            job = QgsMapRendererParallelJob(settings)
            job.start()
            job.waitForFinished()
            rendered = job.renderedImage()
            if rendered.isNull():
                raise ValueError("QGIS 未能渲染所选图层。")

            initial_rgb = QImage(rendered.size(), IMAGE_RGB32)
            initial_rgb.fill(QColor("white"))
            painter = QPainter(initial_rgb)
            painter.drawImage(0, 0, rendered)
            painter.end()
            width, height, gray = image_gray_bytes(initial_rgb)
            threshold = otsu_threshold(gray) if self.threshold_mode.currentIndex() == 0 else self.threshold_value.value()
            # Match the source script: pixels darker than threshold form a white foreground mask.
            mask = bytes(255 if value < threshold else 0 for value in gray)
            gray_preview = gray_image(width, height, gray)
            binary_preview = gray_image(width, height, mask)
            if gray_preview.isNull() or binary_preview.isNull():
                raise ValueError("无法创建灰度或二值预览图像。")
            sizes, counts = box_count(mask, width, height)
            dimension, r_squared, points, slope, intercept = fit_dimension(sizes, counts)

            output_images = {
                "initial_rgb.png": initial_rgb,
                "grayscale.png": gray_preview,
                "binary.png": binary_preview,
            }
            for filename, image in output_images.items():
                if not image.save(os.path.join(outdir, filename), "PNG"):
                    raise IOError("无法写入输出图像：" + filename)
            with open(os.path.join(outdir, "box_counts.csv"), "w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.writer(stream)
                writer.writerow(["box_width_pixels", "occupied_boxes", "log2_box_width", "log2_occupied_boxes"])
                for box_size, count in zip(sizes, counts):
                    writer.writerow([box_size, count, math.log(box_size, 2), math.log(count, 2) if count else ""])
            with open(os.path.join(outdir, "fractal_dimension.txt"), "w", encoding="utf-8") as stream:
                stream.write("地铁线路图箱计数分形维度分析\n")
                stream.write("图层: {}\nCRS: {}\n正方形图像: {} × {} 像素\n".format(layer.name(), layer.crs().authid(), width, height))
                stream.write("二值化阈值: {} ({})\n前景: 灰度值 < 阈值\n".format(threshold, "Otsu" if self.threshold_mode.currentIndex() == 0 else "手动"))
                stream.write("箱宽范围: {} 到 {} 像素\n分形维度: {:.8f}\nR²: {:.8f}\n拟合形式: log2(N) = {:.8f} × log2(box width) + {:.8f}\n".format(sizes[0], sizes[-1], dimension, r_squared, slope, intercept))

            self.images = {"rgb": initial_rgb, "gray": gray_preview, "binary": binary_preview}
            for key, image in self.images.items():
                self._set_preview(key, image)
            self.table.setRowCount(len(sizes))
            for row, (box_size, count) in enumerate(zip(sizes, counts)):
                self.table.setItem(row, 0, QTableWidgetItem(str(box_size)))
                self.table.setItem(row, 1, QTableWidgetItem(str(count)))
            self.plot.set_data(points, slope, intercept)
            self.result_label.setText("分形维度 D = {:.6f}　　R² = {:.6f}　　阈值 = {}".format(dimension, r_squared, threshold))
            self.tabs.setCurrentIndex(3)
            self.status.setText("分析完成。线路前景显示为二值图中的白色区域。结果已保存至：{}".format(outdir))
        except Exception as exc:
            QMessageBox.critical(self, "处理失败", str(exc))
            self.status.setText("处理失败：{}".format(exc))
        finally:
            self.progress.hide()
            self.run_button.setEnabled(True)


class SubwayFractalPlugin:
    def __init__(self, iface):
        self.iface = iface
        self.action = None
        self.dialog = None

    def initGui(self):
        self.action = QAction("地铁线路图分形维度", self.iface.mainWindow())
        self.action.triggered.connect(self.run)
        self.iface.addPluginToMenu("地铁线路图分析", self.action)
        self.iface.addToolBarIcon(self.action)

    def unload(self):
        if self.action:
            self.iface.removePluginMenu("地铁线路图分析", self.action)
            self.iface.removeToolBarIcon(self.action)

    def run(self):
        self.dialog = AnalysisDialog(self.iface, self.iface.mainWindow())
        execute = getattr(self.dialog, "exec", None) or getattr(self.dialog, "exec_")
        execute()


