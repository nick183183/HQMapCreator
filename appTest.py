import sys
import time
import threading
import traceback
import ctypes
import ctypes.wintypes
import numpy as np
import cv2
import mss
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtCore import Qt, QRect, pyqtSignal, QObject
from PyQt5.QtGui import QPainter, QPen, QColor, QPixmap

user32 = ctypes.windll.user32


class Win32HotkeySignals(QObject):
    capture_requested = pyqtSignal()
    exit_requested = pyqtSignal()
    save_requested = pyqtSignal()
    select_region_requested = pyqtSignal()
    undo_requested = pyqtSignal()  # ← F4


class Win32HotkeyListener(threading.Thread):
    VK_F4 = 0x73  # ← новое
    VK_F6 = 0x75
    VK_F7 = 0x76
    VK_F8 = 0x77
    VK_F9 = 0x78
    MOD_NONE = 0x0000

    def __init__(self, signals: Win32HotkeySignals):
        super().__init__(daemon=True)
        self.signals = signals
        self.running = False
        self.hotkey_ids = {
            1: self.VK_F4,  # ← новое: откат
            2: self.VK_F6,
            3: self.VK_F7,
            4: self.VK_F8,
            5: self.VK_F9,
        }

    def run(self):
        for hk_id, vk in self.hotkey_ids.items():
            result = user32.RegisterHotKey(None, hk_id, self.MOD_NONE, vk)
            if not result:
                print(f"Ошибка регистрации хоткея ID={hk_id}")

        self.running = True
        msg = ctypes.wintypes.MSG()
        try:
            while self.running:
                ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if ret == 0 or ret == -1:
                    break

                if msg.message == 0x0312:
                    hk_id = msg.wParam
                    if hk_id == 1:
                        self.signals.undo_requested.emit()
                    elif hk_id == 2:
                        self.signals.capture_requested.emit()
                    elif hk_id == 3:
                        self.signals.exit_requested.emit()
                    elif hk_id == 4:
                        self.signals.save_requested.emit()
                    elif hk_id == 5:
                        self.signals.select_region_requested.emit()
        except Exception as e:
            print(f"Ошибка hotkey listener: {e}")
        finally:
            for hk_id in self.hotkey_ids:
                user32.UnregisterHotKey(None, hk_id)

    def stop(self):
        self.running = False
        user32.PostQuitMessage(0)


class RegionSelector(QWidget):
    region_selected = pyqtSignal(tuple)
    cancelled = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setCursor(Qt.CrossCursor)

        screen = QApplication.primaryScreen().geometry()
        self.setGeometry(screen)

        self.start_pos = None
        self.end_pos = None
        self.drawing = False

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0, 100))

        if self.start_pos and self.end_pos:
            rect = QRect(self.start_pos, self.end_pos).normalized()
            painter.fillRect(rect, QColor(0, 0, 0, 0))
            pen = QPen(QColor(0, 255, 0), 2)
            painter.setPen(pen)
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
                self.region_selected.emit((rect.x(), rect.y(), rect.width(), rect.height()))
            self.close()

    def keyPressEvent(self, event):
        if event.key() == Qt.Key_Escape:
            self.cancelled.emit()
            self.close()


class PreviewWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint |
                            Qt.Tool | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(220, 160)

        screen = QApplication.primaryScreen().geometry()
        self.move(screen.width() - 480, 20)

        self.pixmap = None
        self.drag_pos = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.setBrush(QColor(0, 0, 0, 180))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 15, 15)

        painter.setPen(QColor(255, 255, 255))
        font = painter.font()
        font.setPointSize(9)
        painter.setFont(font)
        painter.drawText(15, 20, "Текущий результат:")

        if self.pixmap:
            preview_rect = QRect(10, 30, 200, 120)
            scaled = self.pixmap.scaled(
                preview_rect.size(),
                Qt.KeepAspectRatio,
                Qt.SmoothTransformation
            )
            x = preview_rect.x() + (preview_rect.width() - scaled.width()) // 2
            y = preview_rect.y() + (preview_rect.height() - scaled.height()) // 2
            painter.drawPixmap(x, y, scaled)
        else:
            painter.setPen(QColor(100, 100, 100))
            painter.drawText(QRect(10, 30, 200, 120), Qt.AlignCenter, "Нет данных")

    def update_image(self, image_path):
        if image_path and Path(image_path).exists():
            self.pixmap = QPixmap(str(image_path))
            self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.drag_pos and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_pos)
            event.accept()


class HUD(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowFlags(Qt.FramelessWindowHint | Qt.WindowStaysOnTopHint |
                            Qt.Tool | Qt.WindowDoesNotAcceptFocus)
        self.setAttribute(Qt.WA_TranslucentBackground)
        self.setFixedSize(220, 175)  # ← чуть выше для новой строки

        screen = QApplication.primaryScreen().geometry()
        self.move(screen.width() - 240, 20)

        self.status = "select area"
        self.frame_count = 0
        self.drag_pos = None

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.setBrush(QColor(0, 0, 0, 180))
        painter.setPen(Qt.NoPen)
        painter.drawRoundedRect(self.rect(), 15, 15)

        colors = {
            "select area": QColor(255, 165, 0),
            "ready": QColor(0, 255, 0),
            "processing": QColor(255, 165, 0),
            "error": QColor(255, 0, 0),
            "first frame": QColor(0, 150, 255),
            "rejected": QColor(255, 80, 80),  # ← новый статус
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

        font.setPointSize(8)
        painter.setFont(font)
        painter.setPen(QColor(180, 180, 180))
        painter.drawText(15, 100, "F9: выбрать область")
        painter.drawText(15, 115, "F6: захват  F4: откат")
        painter.drawText(15, 130, "F7: выход   F8: сохранить")
        painter.drawText(15, 145, "F4 — отменить последний шаг")

    def set_status(self, status):
        self.status = status
        self.update()

    def set_frame_count(self, count):
        self.frame_count = count
        self.update()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_pos = event.globalPos() - self.frameGeometry().topLeft()
            event.accept()

    def mouseMoveEvent(self, event):
        if self.drag_pos and event.buttons() == Qt.LeftButton:
            self.move(event.globalPos() - self.drag_pos)
            event.accept()


class MapStitcherApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)

        self.region = None
        self.current_image = None
        self.hud = None
        self.preview = None
        self.selector = None

        # ← ИСТОРИЯ для отката
        self.history = []  # стек предыдущих current_image
        self.history_size = 2  # по умолчанию

        # Папки
        self.base_dir = Path(__file__).parent
        self.buffer_dir = self.base_dir / "captures"
        self.output_dir = self.base_dir / "output"
        self.buffer_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)

        self.buffer_file = self.buffer_dir / "current.png"

        self.sct = mss.MSS()

        # Загружаем конфиг и stitcher
        try:
            from algoritmTest import ImageStitcher, StitchQualityError
            self.stitcher = ImageStitcher()
            self.StitchQualityError = StitchQualityError
            self.history_size = self.stitcher.__dict__.get('history_size', 2)
            # Читаем history_size из конфига напрямую
            config_path = Path(__file__).parent / "config.json"
            if config_path.exists():
                import json
                with open(config_path, 'r', encoding='utf-8') as f:
                    cfg = json.load(f)
                self.history_size = cfg.get("history_size", 2)
        except ImportError as e:
            print(f"ImageStitcher не найден: {e}")
            self.stitcher = None
            self.StitchQualityError = Exception

        self.hotkey_signals = Win32HotkeySignals()
        self.hotkey_signals.capture_requested.connect(self._capture_and_stitch)
        self.hotkey_signals.exit_requested.connect(self._exit)
        self.hotkey_signals.save_requested.connect(self._manual_save)
        self.hotkey_signals.select_region_requested.connect(self._open_region_selector)
        self.hotkey_signals.undo_requested.connect(self._undo)  # ← F4

        self.hotkey_thread = Win32HotkeyListener(self.hotkey_signals)

    def run(self):
        self.hud = HUD()
        self.hud.set_status("select area")
        self.hud.show()

        self.preview = PreviewWindow()
        self.preview.show()

        self.hotkey_thread.start()
        print("Готово. F9 — область, F6 — захват, F4 — откат, F7 — выход, F8 — сохранить")

        ret = self.app.exec_()
        self.hotkey_thread.stop()
        sys.exit(ret)

    def _push_history(self):
        """Сохраняет текущее состояние в историю перед изменением."""
        if self.current_image is not None:
            self.history.append(self.current_image.copy())
            # Ограничиваем размер истории
            if len(self.history) > self.history_size:
                self.history.pop(0)

    def _undo(self):
        """Откат на шаг назад по F4."""
        if not self.history:
            print("Нечего откатывать — история пуста")
            if self.hud:
                self.hud.set_status("error")
            return

        # Восстанавливаем предыдущее состояние
        self.current_image = self.history.pop()

        if self.hud:
            self.hud.set_status("ready")
            self.hud.set_frame_count(max(0, self.hud.frame_count - 1))

        self._save_to_buffer(self.current_image)
        print(f"Откат выполнен. Осталось в истории: {len(self.history)}")

    def _open_region_selector(self):
        if self.hud:
            self.hud.hide()
        if self.preview:
            self.preview.hide()

        self.selector = RegionSelector()
        self.selector.region_selected.connect(self._on_region_selected)
        self.selector.cancelled.connect(self._on_region_cancelled)
        self.selector.show()

    def _on_region_cancelled(self):
        if self.hud:
            if self.region is None:
                self.hud.set_status("select area")
            self.hud.show()
        if self.preview:
            self.preview.show()

    def _on_region_selected(self, region):
        self.region = {
            "top": region[1],
            "left": region[0],
            "width": region[2],
            "height": region[3]
        }

        if self.hud:
            if self.current_image is None:
                self.hud.set_status("first frame")
            else:
                self.hud.set_status("ready")
            self.hud.show()

        if self.preview:
            self.preview.show()

    def _save_to_buffer(self, image):
        if image is None:
            return

        for f in self.buffer_dir.glob("*.png"):
            if f.name != "current.png":
                try:
                    f.unlink()
                except:
                    pass

        cv2.imwrite(str(self.buffer_file), image)
        self._update_preview()

    def _update_preview(self):
        if self.preview and self.buffer_file.exists():
            self.preview.update_image(self.buffer_file)

    def _copy_to_output(self, prefix="result"):
        if not self.buffer_file.exists():
            print("Нет изображения для сохранения")
            return None

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{prefix}_{timestamp}.png"
        output_file = self.output_dir / filename

        img = cv2.imread(str(self.buffer_file))
        if img is not None:
            cv2.imwrite(str(output_file), img)
            print(f"Сохранено: {output_file}")
            return output_file
        return None

    def _manual_save(self):
        if self.current_image is None:
            print("Нет изображения")
            return
        self._copy_to_output(prefix="manual")

    def _exit(self):
        if self.current_image is not None:
            self._copy_to_output(prefix="final")
        self.app.quit()

    def _capture_and_stitch(self):
        if not self.region:
            print("Область не выбрана. Нажми F9")
            if self.hud:
                self.hud.set_status("select area")
            return

        # Статус "в процессе"
        if self.hud:
            self.hud.set_status("processing")
            self.hud.show()
            self.app.processEvents()
            time.sleep(0.2)

        # Скрываем HUD перед захватом
        if self.hud:
            self.hud.hide()
            self.app.processEvents()
            time.sleep(0.15)

        try:
            screenshot = self.sct.grab(self.region)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            print(f"Ошибка захвата: {e}")
            if self.hud:
                self.hud.set_status("error")
                self.hud.show()
            return

        if self.hud:
            self.hud.show()
            self.app.processEvents()

        if self.current_image is None:
            # Первый кадр — просто сохраняем
            self.current_image = img
            if self.hud:
                self.hud.set_status("ready")
                self.hud.set_frame_count(1)
            self._save_to_buffer(img)
        else:
            if self.stitcher is None:
                print("Stitcher не доступен")
                if self.hud:
                    self.hud.set_status("error")
                return

            try:
                # ← ВАЖНО: сохраняем в историю ПЕРЕД изменением
                self._push_history()

                result, _ = self.stitcher.stitch(self.current_image, img)
                self.current_image = result

                if self.hud:
                    self.hud.set_status("ready")
                    self.hud.set_frame_count(self.hud.frame_count + 1)

                self._save_to_buffer(result)

            except self.StitchQualityError as e:
                # ← ПЛОХОЕ КАЧЕСТВО: откатываем историю и НЕ меняем current_image
                if self.history:
                    self.current_image = self.history.pop()
                print(f"Склейка отклонена: {e}")
                if self.hud:
                    self.hud.set_status("rejected")

            except Exception as e:
                # ← ДРУГАЯ ОШИБКА: тоже откатываем
                if self.history:
                    self.current_image = self.history.pop()
                print(f"Ошибка склейки: {e}")
                if self.hud:
                    self.hud.set_status("error")


if __name__ == "__main__":
    try:
        app = MapStitcherApp()
        app.run()
    except Exception as e:
        print(f"Критическая ошибка: {e}")
        traceback.print_exc()
        input("Нажмите Enter...")