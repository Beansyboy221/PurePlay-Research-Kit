import ctypes.wintypes
import win32gui
import ctypes

from . import base


# --- Ctypes Structs (kept internal to the logic) ---
class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ("dwType", ctypes.wintypes.DWORD),
        ("dwSize", ctypes.wintypes.DWORD),
        ("hDevice", ctypes.wintypes.HANDLE),
        ("wParam", ctypes.wintypes.WPARAM),
    ]


class RAWMOUSE(ctypes.Structure):
    _fields_ = [
        ("usFlags", ctypes.wintypes.USHORT),
        ("ulButtons", ctypes.wintypes.ULONG),
        ("ulRawButtons", ctypes.wintypes.ULONG),
        ("lLastX", ctypes.c_long),
        ("lLastY", ctypes.c_long),
        ("ulExtraInformation", ctypes.wintypes.ULONG),
    ]


class RAWINPUT(ctypes.Structure):
    _fields_ = [("header", RAWINPUTHEADER), ("mouse", RAWMOUSE)]


class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ("usUsagePage", ctypes.wintypes.USHORT),
        ("usUsage", ctypes.wintypes.USHORT),
        ("dwFlags", ctypes.wintypes.DWORD),
        ("hwndTarget", ctypes.wintypes.HWND),
    ]


class RawMouseListener(base.BaseMouseListener):
    def __init__(self):
        super().__init__()
        self._hwnd = None
        self._user32 = ctypes.windll.user32
        self.HWND_MESSAGE = -3
        self.WM_INPUT = 0x00FF
        self.WM_CLOSE = 0x0010
        self.WM_DESTROY = 0x0002
        self.running = False

    def _window_proc(self, hwnd, msg, wparam, lparam):
        """Windows procedure for reading raw input messages."""
        if msg == self.WM_INPUT:
            size = ctypes.wintypes.UINT(0)
            # Get buffer size
            self._user32.GetRawInputData(
                lparam,
                0x10000003,
                None,
                ctypes.byref(size),
                ctypes.sizeof(RAWINPUTHEADER),
            )
            buffer = ctypes.create_string_buffer(size.value)
            # Get actual data
            if (
                self._user32.GetRawInputData(
                    lparam,
                    0x10000003,
                    buffer,
                    ctypes.byref(size),
                    ctypes.sizeof(RAWINPUTHEADER),
                )
                == size.value
            ):
                data = ctypes.cast(buffer, ctypes.POINTER(RAWINPUT)).contents
                if data.header.dwType == 0:  # RIM_TYPEMOUSE
                    with self._lock:
                        self._delta_x += data.mouse.lLastX
                        self._delta_y += data.mouse.lLastY
            return 0

        if msg == self.WM_DESTROY:
            win32gui.PostQuitMessage(0)
            return 0

        return win32gui.DefWindowProc(hwnd, msg, wparam, lparam)

    def _run(self):
        instance_handle = win32gui.GetModuleHandle(None)

        # Register Window Class
        wc = win32gui.WNDCLASS()
        wc.hInstance = instance_handle
        wc.lpszClassName = f"RawMouseInput_{id(self)}"  # Unique class name
        wc.lpfnWndProc = self._window_proc
        class_atom = win32gui.RegisterClass(wc)

        # Create Message-Only Window
        self._hwnd = win32gui.CreateWindow(
            class_atom,
            "RawInputWindow",
            0,
            0,
            0,
            0,
            0,
            self.HWND_MESSAGE,
            0,
            instance_handle,
            None,
        )

        # Register for Raw Input
        rid = RAWINPUTDEVICE()
        rid.usUsagePage = 0x01
        rid.usUsage = 0x02
        rid.dwFlags = 0x00000100  # RIDEV_INPUTSINK
        rid.hwndTarget = self._hwnd

        if not self._user32.RegisterRawInputDevices(
            ctypes.byref(rid), 1, ctypes.sizeof(rid)
        ):
            raise ctypes.WinError()

        self.running = True
        win32gui.PumpMessages()
        self.running = False

    def stop(self):
        if self._hwnd:
            win32gui.PostMessage(self._hwnd, self.WM_CLOSE, 0, 0)
        if self._thread:
            self._thread.join(timeout=1.0)
