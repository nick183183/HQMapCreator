import cv2
from pathlib import Path
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QRect, pyqtSignal
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap

import ctypes

user32 = ctypes.windll.user32
HWND_TOPMOST = -1
SWP_NOMOVE = 0x0002
SWP_NOSIZE = 0x0001


def force_topmost(widget):
    """Принудительно делает окно поверх всех через Win32 API."""
    try:
        hwnd = int(widget.winId())
        user32.SetWindowPos(hwnd, HWND_TOPMOST, 0, 0, 0, 0, SWP_NOMOVE | SWP_NOSIZE)
    except Exception:
        pass


class RegionSelector(QWidget):
    """Окно выбора области экрана."""
    region_selected = pyqtSignal(tuple)
    cancelled = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint |
                            Qt.BypassWindowManagerHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)
        self.setGeometry(QApplication.primaryScreen().geometry())
        self.start_pos = None
        self.end_pos = None
        self.drawing = False

    def show(self):
        super().show()
        force_topmost(self)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))
        if self.start_pos and self.end_pos:
            rect = QRect(self.start_pos, self.end_pos).normalized()
            painter.fillRect(rect, QColor(0, 0, 0, 0))
            painter.setPen(QPen(QColor(0, 255, 0), 2))
            painter.drawRect(rect)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.start_pos = event.pos()
            self.drawing = True

    def mouseMoveEvent(self, event):
        if self.drawing:
            self.end_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.start_pos and self.end_pos:
            rect = QRect(self.start_pos, self.end_pos).normalized()
            if rect.width() > 10 and rect.height() > 10:
                self.drawing = False
                self.region_selected.emit(
                    (rect.x(), rect.y(), rect.width(), rect.height())
                )
            self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.cancelled.emit()
            self.close()


class PreviewWindow(QWidget):
    """Окно превью с ресайзом с сохранением пропорций изображения."""
    clicked = pyqtSignal(int, int)

    MARGIN_LEFT = 10
    MARGIN_TOP = 40
    MARGIN_RIGHT = 10
    MARGIN_BOTTOM = 10
    TITLE_HEIGHT = 30
    RESIZE_MARGIN = 12

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint |
                            Qt.Tool | Qt.WindowDoesNotAcceptFocus |
                            Qt.BypassWindowManagerHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)

        self.setMinimumSize(200, 150)
        self.resize(350, 260)

        screen = QApplication.primaryScreen().geometry()
        self.move(screen.width() - 500, 20)

        self.pixmap = None
        self.image_shape = None
        self.view_rect = None
        self.drag_pos = None
        self.waiting_for_click = False

        self.resizing = False
        self.resize_origin = None
        self.resize_start_pos = None
        self.resize_start_size = None

    def show(self):
        super().show()
        force_topmost(self)

    def showEvent(self, event):
        super().showEvent(event)
        force_topmost(self)

    def set_waiting_for_click(self, enabled):
        self.waiting_for_click = enabled
        if enabled:
            self.setCursor(Qt.CrossCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
        self.update()

    def _get_resize_corner(self, pos):
        """Определяет угол для ресайза."""
        rect = self.rect()
        m = self.RESIZE_MARGIN

        near_right = pos.x() > rect.width() - m
        near_left = pos.x() < m
        near_bottom = pos.y() > rect.height() - m
        near_top = pos.y() < m

        if near_bottom and near_right:
            return "bottom-right"
        if near_bottom and near_left:
            return "bottom-left"
        if near_top and near_right:
            return "top-right"
        if near_top and near_left:
            return "top-left"

        if near_bottom:
            return "bottom-right" if pos.x() > rect.width() / 2 else "bottom-left"
        if near_top:
            return "top-right" if pos.x() > rect.width() / 2 else "top-left"
        if near_right:
            return "bottom-right" if pos.y() > rect.height() / 2 else "top-right"
        if near_left:
            return "bottom-left" if pos.y() > rect.height() / 2 else "top-left"

        return None

    def _get_aspect_ratio(self):
        """Соотношение сторон для окна."""
        if self.image_shape is not None:
            img_h, img_w = self.image_shape
            if img_h > 0 and img_w > 0:
                content_w = img_w
                content_h = img_h
                total_w = content_w + self.MARGIN_LEFT + self.MARGIN_RIGHT
                total_h = content_h + self.MARGIN_TOP + self.MARGIN_BOTTOM
                return total_w / total_h
        return self.width() / max(1, self.height())

    def _apply_resize(self, global_pos):
        """Применяет ресайз с сохранением пропорций."""
        if not self.resize_origin or not self.resize_start_pos:
            return

        delta = global_pos - self.resize_start_pos
        aspect = self._get_aspect_ratio()

        start_w, start_h = self.resize_start_size.width(), self.resize_start_size.height()
        new_w, new_h = start_w, start_h

        if "right" in self.resize_origin:
            new_w = start_w + delta.x()
        elif "left" in self.resize_origin:
            new_w = start_w - delta.x()

        if "bottom" in self.resize_origin:
            new_h = start_h + delta.y()
        elif "top" in self.resize_origin:
            new_h = start_h - delta.y()

        dw = abs(new_w - start_w)
        dh = abs(new_h - start_h)

        if dw > dh:
            new_h = int(new_w / aspect)
        else:
            new_w = int(new_h * aspect)

        min_w = self.minimumWidth()
        min_h = self.minimumHeight()
        if new_w < min_w:
            new_w = min_w
            new_h = int(new_w / aspect)
        if new_h < min_h:
            new_h = min_h
            new_w = int(new_h * aspect)

        new_x = self.x()
        new_y = self.y()
        if "left" in self.resize_origin:
            new_x = self._resize_window_pos.x() + (start_w - new_w)
        if "top" in self.resize_origin:
            new_y = self._resize_window_pos.y() + (start_h - new_h)

        self.resize(new_w, new_h)
        self.move(new_x, new_y)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.waiting_for_click:
                self.clicked.emit(event.pos().x(), event.pos().y())
                event.accept()
                return

            corner = self._get_resize_corner(event.pos())
            if corner:
                self.resizing = True
                self.resize_origin = corner
                self.resize_start_pos = event.globalPos()
                self.resize_start_size = self.size()
                self._resize_window_pos = self.pos()
                event.accept()
                return

            self.drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.resizing:
            self._apply_resize(event.globalPos())
            event.accept()
            return

        if not self.waiting_for_click:
            corner = self._get_resize_corner(event.pos())
            if corner:
                if "top-left" in corner or "bottom-right" in corner:
                    self.setCursor(Qt.SizeFDiagCursor)
                else:
                    self.setCursor(Qt.SizeBDiagCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

        if self.drag_pos and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_pos)
            event.accept()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.resizing = False
            self.resize_origin = None
            event.accept()

    def _get_preview_rect(self):
        """Вычисляет область для изображения."""
        return QRect(
            self.MARGIN_LEFT,
            self.MARGIN_TOP,
            max(1, self.width() - self.MARGIN_LEFT - self.MARGIN_RIGHT),
            max(1, self.height() - self.MARGIN_TOP - self.MARGIN_BOTTOM)
        )

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.setBrush(QColor(0, 0, 0, 180))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 15, 15)

        painter.setPen(QPen(QColor(80, 80, 80), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(self.rect().adjusted(2, 2, -2, -2))

        painter.setPen(QColor(255, 255, 255))
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        title = "Кликни по центру:" if self.waiting_for_click else "Текущий результат:"
        painter.drawText(15, 25, title)

        preview_rect = self._get_preview_rect()

        if self.pixmap:
            scaled = self.pixmap.scaled(
                preview_rect.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            x = preview_rect.x() + (preview_rect.width() - scaled.width()) // 2
            y = preview_rect.y() + (preview_rect.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)

            if self.view_rect and self.image_shape:
                self._draw_view_rect(painter, scaled, x, y)

            if self.waiting_for_click:
                painter.setPen(QPen(QColor(255, 200, 0), 2, Qt.DashLine))
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(x, y, scaled.width(), scaled.height())
        else:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(preview_rect, Qt.AlignCenter, "Нет данных")

        if not self.waiting_for_click:
            painter.setPen(QPen(QColor(150, 150, 150), 1))
            corner_size = 8
            w = self.width()
            h = self.height()
            painter.drawLine(w - corner_size - 4, h - 4, w - 4, h - corner_size - 4)
            painter.drawLine(w - corner_size * 2 - 4, h - 4, w - 4, h - corner_size * 2 - 4)

    def _draw_view_rect(self, painter, scaled_pixmap, sx, sy):
        try:
            img_h, img_w = self.image_shape
            if img_w == 0 or img_h == 0:
                return
            scale_x = scaled_pixmap.width() / img_w
            scale_y = scaled_pixmap.height() / img_h
            rx, ry, rw, rh = self.view_rect
            painter.setPen(QPen(QColor(255, 165, 0), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(
                int(sx + rx * scale_x), int(sy + ry * scale_y),
                int(rw * scale_x), int(rh * scale_y)
            )
        except Exception:
            pass

    def update_image(self, image_path):
        if image_path and Path(image_path).exists():
            self.pixmap = QPixmap(str(image_path))
            img = cv2.imread(str(image_path))
            if img is not None:
                self.image_shape = img.shape[:2]
            self.update()

    def set_view_rect(self, view_rect, image_shape):
        self.view_rect = view_rect
        self.image_shape = image_shape
        self.update()

    def clear_view_rect(self):
        self.view_rect = None
        self.update()

    def widget_to_image_coords(self, widget_x, widget_y):
        """Пересчитывает координаты клика в виджете в координаты оригинального изображения."""
        if self.image_shape is None:
            return None

        img_h, img_w = self.image_shape
        if img_w == 0 or img_h == 0:
            return None

        preview_rect = self._get_preview_rect()

        scale_x = preview_rect.width() / img_w
        scale_y = preview_rect.height() / img_h
        scale = min(scale_x, scale_y)

        scaled_w = int(img_w * scale)
        scaled_h = int(img_h * scale)

        offset_x = preview_rect.x() + (preview_rect.width() - scaled_w) // 2
        offset_y = preview_rect.y() + (preview_rect.height() - scaled_h) // 2

        local_x = widget_x - offset_x
        local_y = widget_y - offset_y

        if local_x < 0 or local_x >= scaled_w or local_y < 0 or local_y >= scaled_h:
            return None

        img_x = int(local_x / scale)
        img_y = int(local_y / scale)

        return (img_x, img_y)


class HUD(QWidget):
    """Индикатор статуса приложения."""

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint |
                            Qt.Tool | Qt.WindowDoesNotAcceptFocus |
                            Qt.BypassWindowManagerHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setAttribute(Qt.WA_ShowWithoutActivating)
        self.setFixedSize(220, 235)  # ← увеличена высота для нового индикатора
        screen = QApplication.primaryScreen().geometry()
        self.move(screen.width() - 240, 20)
        self.status = "select area"
        self.frame_count = 0
        self.tracking_enabled = False
        self.strict_mode = True  # ← новое: строгий режим по умолчанию
        self.drag_pos = None

    def show(self):
        super().show()
        force_topmost(self)

    def showEvent(self, event):
        super().showEvent(event)
        force_topmost(self)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setBrush(QColor(0, 0, 0, 180))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 15, 15)

        colors = {
            "select area": QColor(255, 165, 0),
            "loaded": QColor(0, 200, 255),
            "ready": QColor(0, 255, 0),
            "processing": QColor(255, 165, 0),
            "error": QColor(255, 0, 0),
            "rejected": QColor(255, 80, 80),
            "click on preview": QColor(255, 200, 0),
            "enable tracking": QColor(255, 100, 100),
            "load first": QColor(255, 150, 0),
            "select area first": QColor(255, 150, 0),
        }
        painter.setBrush(colors.get(self.status, QColor(128, 128, 128)))
        painter.drawEllipse(15, 15, 30, 30)

        painter.setPen(QColor(255, 255, 255))
        font = painter.font()
        font.setPointSize(10)
        painter.setFont(font)
        painter.drawText(55, 35, self.status)

        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(15, 75, f"Кадров: {self.frame_count}")

        # Индикатор трекинга
        track_color = QColor(0, 255, 0) if self.tracking_enabled else QColor(100, 100, 100)
        painter.setBrush(track_color)
        painter.drawEllipse(150, 65, 10, 10)
        painter.setPen(QColor(180, 180, 180))
        font.setPointSize(7)
        painter.setFont(font)
        painter.drawText(165, 75, "трекинг")

        # ← НОВОЕ: индикатор строгого режима
        strict_color = QColor(255, 100, 100) if self.strict_mode else QColor(100, 100, 100)
        painter.setBrush(strict_color)
        painter.drawEllipse(15, 85, 10, 10)
        painter.setPen(QColor(180, 180, 180))
        font.setPointSize(7)
        painter.setFont(font)
        painter.drawText(30, 95, "строгий" if self.strict_mode else "обычный")

        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QColor(180, 180, 180))
        painter.drawText(15, 115, "F1: трекинг")
        painter.drawText(15, 130, "F2: поиск центра")
        painter.drawText(15, 145, "F3: строгий режим")
        painter.drawText(15, 160, "F5: загрузить")
        painter.drawText(15, 175, "F6: захват  F4: откат")
        painter.drawText(15, 190, "F7: выход   F8: сохранить")
        painter.drawText(15, 205, "F9: выбрать область")

    def set_status(self, status):
        self.status = status
        self.update()

    def set_frame_count(self, count):
        self.frame_count = count
        self.update()

    def set_tracking(self, enabled):
        self.tracking_enabled = enabled
        self.update()

    def set_strict_mode(self, enabled):
        """Устанавливает строгий режим."""
        self.strict_mode = enabled
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.drag_pos and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_pos)
            event.accept()