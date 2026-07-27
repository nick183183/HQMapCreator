import sys
import time
import json
import traceback
import numpy as np
import cv2
import mss
from pathlib import Path
from datetime import datetime
from PyQt5.QtWidgets import QApplication, QFileDialog
from PyQt5.QtCore import QTimer

from stitcher import ImageStitcher, StitchQualityError
from hotkeys import HotkeySignals, HotkeyListener, MouseTrackerSignals, MouseTracker
from ui import HUD, PreviewWindow, RegionSelector, force_topmost


class MapStitcherApp:
    def __init__(self):
        self.app = QApplication(sys.argv)
        self.app.setQuitOnLastWindowClosed(False)

        self.region = None
        self.current_image = None
        self.hud = None
        self.preview = None
        self.selector = None

        self.history = []
        self.history_size = 2

        self.tracking_enabled = False
        self.current_view_rect = None
        self.accumulated_dx = 0
        self.accumulated_dy = 0
        self.searching_position = False

        # ← НОВОЕ: строгий режим (по умолчанию включён)
        self.strict_mode = True

        self.base_dir = Path(__file__).parent
        self.buffer_dir = self.base_dir / "captures"
        self.output_dir = self.base_dir / "output"
        self.buffer_dir.mkdir(exist_ok=True)
        self.output_dir.mkdir(exist_ok=True)
        self.buffer_file = self.buffer_dir / "current.png"

        self.sct = mss.MSS()

        try:
            self.stitcher = ImageStitcher()
            config_path = self.base_dir / "config.json"
            if config_path.exists():
                with open(config_path, 'r', encoding='utf-8') as f:
                    self.history_size = json.load(f).get("history_size", 2)
        except ImportError as e:
            print(f"ImageStitcher не найден: {e}")
            self.stitcher = None

        self.hotkey_signals = HotkeySignals()
        self.hotkey_signals.capture_requested.connect(self._capture_and_stitch)
        self.hotkey_signals.exit_requested.connect(self._exit)
        self.hotkey_signals.save_requested.connect(self._manual_save)
        self.hotkey_signals.select_region_requested.connect(self._open_region_selector)
        self.hotkey_signals.undo_requested.connect(self._undo)
        self.hotkey_signals.load_first_requested.connect(self._load_first_image)
        self.hotkey_signals.toggle_tracking_requested.connect(self._toggle_tracking)
        self.hotkey_signals.start_position_search_requested.connect(self._start_position_search)
        self.hotkey_signals.toggle_strict_mode_requested.connect(self._toggle_strict_mode)  # ← новое

        self.hotkey_thread = HotkeyListener(self.hotkey_signals)

        self.mouse_signals = MouseTrackerSignals()
        self.mouse_signals.mouse_moved.connect(self._on_mouse_moved)
        self.mouse_tracker = MouseTracker(self.mouse_signals)

        self.topmost_timer = QTimer()
        self.topmost_timer.timeout.connect(self._force_all_topmost)
        self.topmost_timer.start(500)

    def _force_all_topmost(self):
        if self.hud and self.hud.isVisible():
            force_topmost(self.hud)
        if self.preview and self.preview.isVisible():
            force_topmost(self.preview)

    def run(self):
        self.hud = HUD()
        self.hud.set_status("select area")
        self.hud.set_strict_mode(self.strict_mode)  # ← новое
        self.hud.show()

        self.preview = PreviewWindow()
        self.preview.clicked.connect(self._on_preview_clicked)
        self.preview.show()

        self.hotkey_thread.start()
        self.mouse_tracker.start()

        print(
            "Готово. F1-трекинг F2-поиск центра F3-строгий режим F5-загрузить F6-захват F4-откат F7-выход F8-сохранить F9-область")
        ret = self.app.exec_()
        self.hotkey_thread.stop()
        self.mouse_tracker.stop()
        sys.exit(ret)

    def _toggle_strict_mode(self):
        """F3: переключение строгого режима."""
        self.strict_mode = not self.strict_mode
        if self.hud:
            self.hud.set_strict_mode(self.strict_mode)
            force_topmost(self.hud)
        print(f"Строгий режим {'ВКЛ' if self.strict_mode else 'ВЫКЛ'}")

    def _on_preview_clicked(self, widget_x, widget_y):
        """Обработка клика по превью в режиме поиска позиции."""
        if not self.searching_position:
            return

        if self.preview is None:
            return

        img_coords = self.preview.widget_to_image_coords(widget_x, widget_y)
        if img_coords is None:
            print("Клик вне области изображения")
            return

        center_x, center_y = img_coords
        print(f"Клик в координатах изображения: ({center_x}, {center_y})")

        region_w = self.region["width"]
        region_h = self.region["height"]

        rect_x = center_x - region_w // 2
        rect_y = center_y - region_h // 2

        self.current_view_rect = (rect_x, rect_y, region_w, region_h)
        self.searching_position = False

        if self.preview:
            self.preview.set_waiting_for_click(False)
            self.preview.set_view_rect(self.current_view_rect, self.current_image.shape[:2])

        if self.hud:
            self.hud.set_status("ready")
            self.hud.set_frame_count(1)
            force_topmost(self.hud)

        print(f"Позиция установлена: {self.current_view_rect}")
        print("Теперь можно склеивать (F6)")

    def _toggle_tracking(self):
        self.tracking_enabled = not self.tracking_enabled
        self.mouse_tracker.set_enabled(self.tracking_enabled)
        if self.hud:
            self.hud.set_tracking(self.tracking_enabled)
            force_topmost(self.hud)
        print(f"Трекинг {'ВКЛ' if self.tracking_enabled else 'ВЫКЛ'}")
        if not self.tracking_enabled and self.preview:
            self.preview.clear_view_rect()

    def _on_mouse_moved(self, dx, dy):
        try:
            if not self.tracking_enabled or not self.current_view_rect or self.current_image is None:
                return
            self.accumulated_dx -= dx
            self.accumulated_dy -= dy
            x, y, w, h = self.current_view_rect
            self.current_view_rect = (x - dx, y - dy, w, h)
            if self.preview:
                self.preview.set_view_rect(self.current_view_rect, self.current_image.shape[:2])
        except Exception:
            pass

    def _compute_search_region(self):
        if not self.current_view_rect or self.current_image is None:
            return None

        x, y, w, h = self.current_view_rect
        padding_ratio = 0.5
        if self.stitcher is not None:
            padding_ratio = getattr(self.stitcher, 'search_padding_ratio', 0.5)

        pad_x, pad_y = int(w * padding_ratio), int(h * padding_ratio)

        return (
            int(x + self.accumulated_dx - pad_x),
            int(y + self.accumulated_dy - pad_y),
            int(w + 2 * pad_x),
            int(h + 2 * pad_y)
        )

    def _load_first_image(self):
        """F5: загрузка изображения с диска."""
        file_path, _ = QFileDialog.getOpenFileName(
            None, "Выберите первый снимок", str(self.base_dir),
            "Изображения (*.png *.jpg *.jpeg *.bmp);;Все файлы (*.*)"
        )
        if not file_path:
            return
        img = cv2.imread(file_path)
        if img is None:
            print(f"Не удалось загрузить: {file_path}")
            if self.hud:
                self.hud.set_status("error")
            return

        self.current_image = img
        self.history.clear()
        self.current_view_rect = None
        self.accumulated_dx = 0
        self.accumulated_dy = 0

        if self.hud:
            self.hud.set_status("loaded")
            self.hud.set_frame_count(0)
            force_topmost(self.hud)

        self._save_to_buffer(img)
        if self.preview:
            self.preview.clear_view_rect()
            self.preview.set_waiting_for_click(False)

        print(f"Загружено: {file_path}")
        print("Используй F9 для выбора области, F1 для трекинга, F2 для поиска центра")

    def _start_position_search(self):
        """F2: запуск режима поиска центра."""
        if self.current_image is None:
            print("Сначала загрузи изображение (F5)")
            if self.hud:
                self.hud.set_status("load first")
            return

        if self.region is None:
            print("Сначала выбери область (F9)")
            if self.hud:
                self.hud.set_status("select area first")
            return

        self.searching_position = True

        if self.hud:
            self.hud.set_status("click on preview")
            force_topmost(self.hud)

        if self.preview:
            self.preview.set_waiting_for_click(True)

        print("Режим поиска: кликни по превью где центр твоего изображения")

    def _push_history(self):
        if self.current_image is not None:
            self.history.append(self.current_image.copy())
            if len(self.history) > self.history_size:
                self.history.pop(0)

    def _undo(self):
        if not self.history:
            print("Нечего откатывать")
            if self.hud:
                self.hud.set_status("error")
            return
        self.current_image = self.history.pop()
        self.current_view_rect = None
        self.accumulated_dx = 0
        self.accumulated_dy = 0
        if self.hud:
            self.hud.set_status("ready")
            self.hud.set_frame_count(max(0, self.hud.frame_count - 1))
            force_topmost(self.hud)
        self._save_to_buffer(self.current_image)
        if self.preview:
            self.preview.clear_view_rect()
            self.preview.set_waiting_for_click(False)
        print(f"Откат. В истории: {len(self.history)}")

    def _open_region_selector(self):
        """F9: выбор области на экране."""
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
            force_topmost(self.hud)
        if self.preview:
            self.preview.show()
            force_topmost(self.preview)

    def _on_region_selected(self, region):
        self.region = {
            "top": region[1], "left": region[0],
            "width": region[2], "height": region[3]
        }
        if self.hud:
            if self.current_image is None:
                self.hud.set_status("loaded")
            else:
                self.hud.set_status("ready")
            self.hud.show()
            force_topmost(self.hud)
        if self.preview:
            self.preview.show()
            force_topmost(self.preview)

    def _save_to_buffer(self, image):
        if image is None:
            return
        for f in self.buffer_dir.glob("*.png"):
            if f.name != "current.png":
                try:
                    f.unlink()
                except Exception:
                    pass
        cv2.imwrite(str(self.buffer_file), image)
        if self.preview and self.buffer_file.exists():
            self.preview.update_image(self.buffer_file)

    def _copy_to_output(self, prefix="result"):
        if not self.buffer_file.exists():
            return None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = self.output_dir / f"{prefix}_{timestamp}.png"
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
        # Проверка: область выбрана?
        if not self.region:
            print("Область не выбрана. Нажми F9")
            if self.hud:
                self.hud.set_status("select area")
                force_topmost(self.hud)
            return

        # Проверка: трекинг включён?
        if not self.tracking_enabled:
            print("Трекинг выключен. Нажми F1")
            if self.hud:
                self.hud.set_status("enable tracking")
                force_topmost(self.hud)
            return

        # Режим поиска позиции — ждём клика по превью
        if self.searching_position:
            print("Режим поиска: кликни по превью где центр твоего изображения")
            return

        # Скрываем ВЕСЬ UI перед захватом
        hud_was_visible = self.hud and self.hud.isVisible()
        preview_was_visible = self.preview and self.preview.isVisible()

        if self.hud:
            self.hud.hide()
        if self.preview:
            self.preview.hide()

        self.app.processEvents()
        time.sleep(0.2)

        try:
            screenshot = self.sct.grab(self.region)
            img = np.array(screenshot)
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        except Exception as e:
            print(f"Ошибка захвата: {e}")
            if hud_was_visible and self.hud:
                self.hud.set_status("error")
                self.hud.show()
                force_topmost(self.hud)
            if preview_was_visible and self.preview:
                self.preview.show()
                force_topmost(self.preview)
            return

        if hud_was_visible and self.hud:
            self.hud.set_status("processing")
            self.hud.show()
            force_topmost(self.hud)
            self.app.processEvents()

        if preview_was_visible and self.preview:
            self.preview.show()
            force_topmost(self.preview)
            self.app.processEvents()

        if self.current_image is None:
            self.current_image = img
            self.current_view_rect = (0, 0, img.shape[1], img.shape[0])
            self.accumulated_dx = 0
            self.accumulated_dy = 0
            if self.hud:
                self.hud.set_status("ready")
                self.hud.set_frame_count(1)
                force_topmost(self.hud)
            self._save_to_buffer(img)
        else:
            if self.stitcher is None:
                if self.hud:
                    self.hud.set_status("error")
                    force_topmost(self.hud)
                return
            try:
                self._push_history()
                search_region = None
                if self.tracking_enabled and self.current_view_rect:
                    search_region = self._compute_search_region()

                # ← ПЕРЕДАЁМ strict_mode в stitcher
                result, _, last_capture_rect, used_region = self.stitcher.stitch(
                    self.current_image, img,
                    use_cache=True,
                    search_region=search_region,
                    strict_mode=self.strict_mode
                )
                self.current_image = result
                self.current_view_rect = last_capture_rect
                self.accumulated_dx = 0
                self.accumulated_dy = 0

                if self.hud:
                    self.hud.set_status("ready")
                    self.hud.set_frame_count(self.hud.frame_count + 1)
                    force_topmost(self.hud)
                self._save_to_buffer(result)

                if search_region is not None:
                    mode = "область" if used_region else "fallback"
                    print(f"Режим: {mode}")

            except StitchQualityError as e:
                if self.history:
                    self.current_image = self.history.pop()
                self.accumulated_dx = 0
                self.accumulated_dy = 0
                print(f"Склейка отклонена: {e}")
                if self.hud:
                    self.hud.set_status("rejected")
                    force_topmost(self.hud)

            except Exception as e:
                if self.history:
                    self.current_image = self.history.pop()
                self.accumulated_dx = 0
                self.accumulated_dy = 0
                print(f"Ошибка склейки: {e}")
                if self.hud:
                    self.hud.set_status("error")
                    force_topmost(self.hud)