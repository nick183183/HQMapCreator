import time
import threading
import ctypes
import ctypes.wintypes
from PyQt5.QtCore import pyqtSignal, QObject


user32 = ctypes.windll.user32
VK_LBUTTON = 0x01


class HotkeySignals(QObject):
    capture_requested = pyqtSignal()
    exit_requested = pyqtSignal()
    save_requested = pyqtSignal()
    select_region_requested = pyqtSignal()
    undo_requested = pyqtSignal()
    load_first_requested = pyqtSignal()
    toggle_tracking_requested = pyqtSignal()
    start_position_search_requested = pyqtSignal()
    toggle_strict_mode_requested = pyqtSignal()  # ← новое: F3


class HotkeyListener(threading.Thread):
    VK_F1 = 0x70
    VK_F2 = 0x71
    VK_F3 = 0x72  # ← новое: строгий режим
    VK_F4 = 0x73
    VK_F5 = 0x74
    VK_F6 = 0x75
    VK_F7 = 0x76
    VK_F8 = 0x77
    VK_F9 = 0x78
    MOD_NONE = 0x0000

    def __init__(self, signals: HotkeySignals):
        super().__init__(daemon=True)
        self.signals = signals
        self.running = False
        self.hotkey_ids = {
            1: self.VK_F1,
            2: self.VK_F2,
            3: self.VK_F3,  # ← новое
            4: self.VK_F4,
            5: self.VK_F5,
            6: self.VK_F6,
            7: self.VK_F7,
            8: self.VK_F8,
            9: self.VK_F9,
        }

    def run(self):
        for hk_id, vk in self.hotkey_ids.items():
            if not user32.RegisterHotKey(None, hk_id, self.MOD_NONE, vk):
                print(f"Ошибка регистрации хоткея ID={hk_id}")

        self.running = True
        msg = ctypes.wintypes.MSG()
        try:
            while self.running:
                ret = user32.GetMessageW(ctypes.byref(msg), None, 0, 0)
                if ret == 0 or ret == -1:
                    break
                if msg.message == 0x0312:
                    self._dispatch(msg.wParam)
        except Exception as e:
            print(f"Ошибка hotkey listener: {e}")
        finally:
            for hk_id in self.hotkey_ids:
                user32.UnregisterHotKey(None, hk_id)

    def _dispatch(self, hk_id):
        dispatch = {
            1: self.signals.toggle_tracking_requested,
            2: self.signals.start_position_search_requested,
            3: self.signals.toggle_strict_mode_requested,  # ← новое
            4: self.signals.undo_requested,
            5: self.signals.load_first_requested,
            6: self.signals.capture_requested,
            7: self.signals.exit_requested,
            8: self.signals.save_requested,
            9: self.signals.select_region_requested,
        }
        signal = dispatch.get(hk_id)
        if signal:
            signal.emit()

    def stop(self):
        self.running = False
        user32.PostQuitMessage(0)


class MouseTrackerSignals(QObject):
    mouse_moved = pyqtSignal(int, int)


class MouseTracker(threading.Thread):
    """Отслеживает движение мыши при зажатой ЛКМ."""

    def __init__(self, signals: MouseTrackerSignals):
        super().__init__(daemon=True)
        self.signals = signals
        self.running = False
        self.enabled = False
        self.last_pos = None

    def run(self):
        self.running = True
        while self.running:
            try:
                if not self.enabled:
                    time.sleep(0.1)
                    continue

                key_state = user32.GetAsyncKeyState(VK_LBUTTON)
                if (key_state & 0x8000) != 0:
                    pt = ctypes.wintypes.POINT()
                    if user32.GetCursorPos(ctypes.byref(pt)):
                        current_pos = (pt.x, pt.y)
                        if self.last_pos is not None:
                            dx = current_pos[0] - self.last_pos[0]
                            dy = current_pos[1] - self.last_pos[1]
                            if dx != 0 or dy != 0:
                                self.signals.mouse_moved.emit(dx, dy)
                        self.last_pos = current_pos
                else:
                    self.last_pos = None

                time.sleep(0.01)
            except Exception:
                time.sleep(0.1)

    def stop(self):
        self.running = False

    def set_enabled(self, enabled):
        self.enabled = enabled
        if not enabled:
            self.last_pos = None