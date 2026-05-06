import threading
import select
import evdev

from . import base


class RawMouseListener(base.BaseMouseListener):
    def __init__(self):
        super().__init__()
        self._stop_event = threading.Event()

    def _run(self):
        devices = [evdev.InputDevice(path) for path in evdev.list_devices()]
        mouse_device = next(
            (
                device
                for device in devices
                if evdev.ecodes.EV_REL in device.capabilities()
            ),
            None,
        )

        if not mouse_device:
            return

        mouse_device.grab()
        self.running = True
        try:
            while not self._stop_event.is_set():
                r, _, _ = select.select([mouse_device.fd], [], [], 0.1)
                if r:
                    for event in mouse_device.read():
                        if event.type == evdev.ecodes.EV_REL:
                            with self._lock:
                                if event.code == evdev.ecodes.REL_X:
                                    self._delta_x += event.value
                                elif event.code == evdev.ecodes.REL_Y:
                                    self._delta_y += event.value
        finally:
            self.running = False
            mouse_device.ungrab()
            mouse_device.close()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=1.0)
