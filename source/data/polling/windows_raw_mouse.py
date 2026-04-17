import ctypes.wintypes
import threading
import win32gui

# Are there any external modules that do this?

mouse_deltas = [0, 0]  # [delta_x, delta_y]
mouse_lock = threading.Lock()

#region ctypes Structs
class RAWINPUTHEADER(ctypes.Structure):
    _fields_ = [
        ('dwType', ctypes.wintypes.DWORD),
        ('dwSize', ctypes.wintypes.DWORD),
        ('hDevice', ctypes.wintypes.HANDLE),
        ('wParam', ctypes.wintypes.WPARAM)
    ]

class RAWMOUSE(ctypes.Structure):
    _fields_ = [
        ('usFlags', ctypes.wintypes.USHORT),
        ('ulButtons', ctypes.wintypes.ULONG),
        ('ulRawButtons', ctypes.wintypes.ULONG),
        ('lLastX', ctypes.c_long),
        ('lLastY', ctypes.c_long),
        ('ulExtraInformation', ctypes.wintypes.ULONG)
    ]

class RAWINPUT(ctypes.Structure):
    _fields_ = [
        ('header', RAWINPUTHEADER),
        ('mouse',  RAWMOUSE)
    ]

class RAWINPUTDEVICE(ctypes.Structure):
    _fields_ = [
        ('usUsagePage', ctypes.wintypes.USHORT),
        ('usUsage', ctypes.wintypes.USHORT),
        ('dwFlags', ctypes.wintypes.DWORD),
        ('hwndTarget', ctypes.wintypes.HWND)
    ]
#endregion

#region Constants
USER32_LIBRARY = ctypes.windll.user32
HWND_MESSAGE = -3
WM_INPUT = 0x00FF
WM_CLOSE = 0x0010
WM_DESTROY = 0x0002
#endregion

#region Mouse Listener
def listen_for_mouse_movement(kill_event: threading.Event) -> None:
    '''Listens for raw mouse input and updates global mouse delta values.'''
    instance_handle = win32gui.GetModuleHandle(None)
    class_name = 'RawInputWindow'
    window = win32gui.WNDCLASS()
    window.hInstance = instance_handle
    window.lpszClassName = class_name
    window.lpfnWndProc = _raw_input_window_procedure
    win32gui.RegisterClass(window)
    window_handle = win32gui.CreateWindow(
        class_name,                     # lpClassName
        'Raw Input Hidden Window',      # lpWindowName
        0,                              # dwStyle
        0, 0, 0, 0,                     # x, y, width, height
        HWND_MESSAGE,                   # hWndParent (message-only window)
        0,                              # hMenu
        instance_handle,                # hInstance
        None                            # lpParam
    )

    device = RAWINPUTDEVICE()
    device.usUsagePage = 0x01   # Generic Desktop Controls
    device.usUsage = 0x02       # Mouse
    device.dwFlags = 0x00000100 # RIDEV_INPUTSINK: receive input even when unfocused
    device.hwndTarget = window_handle
    if not USER32_LIBRARY.RegisterRawInputDevices(ctypes.byref(device), 1, ctypes.sizeof(device)):
        raise ctypes.WinError(descr='Failed to register raw input device.')

    def kill_watcher():
        kill_event.wait()
        win32gui.PostMessage(window_handle, WM_CLOSE, 0, 0)
    threading.Thread(target=kill_watcher, daemon=True).start()
    win32gui.PumpMessages()

def _raw_input_window_procedure(
        window_handle: ctypes.wintypes.HWND, 
        message: ctypes.wintypes.UINT, 
        input_code: ctypes.wintypes.WPARAM, 
        data_handle: ctypes.wintypes.LPARAM
    ) -> ctypes.c_long:
    '''Window procedure for processing raw input messages. Updates mouse deltas on WM_INPUT and exits on WM_DESTROY.'''
    if message == WM_INPUT:
        buffer_size = ctypes.wintypes.UINT(0)
        if USER32_LIBRARY.GetRawInputData(data_handle, 0x10000003, None, ctypes.byref(buffer_size), ctypes.sizeof(RAWINPUTHEADER)) == 0:
            buffer = ctypes.create_string_buffer(buffer_size.value)
            if USER32_LIBRARY.GetRawInputData(data_handle, 0x10000003, buffer, ctypes.byref(buffer_size), ctypes.sizeof(RAWINPUTHEADER)) == buffer_size.value:
                raw_input_data = ctypes.cast(buffer, ctypes.POINTER(RAWINPUT)).contents
                if raw_input_data.header.dwType == 0:
                    with mouse_lock:
                        mouse_deltas[0] += raw_input_data.mouse.lLastX
                        mouse_deltas[1] += raw_input_data.mouse.lLastY
        return 0
    elif message == WM_DESTROY:
        win32gui.PostQuitMessage(0)
        return 0
    return win32gui.DefWindowProc(window_handle, message, input_code, data_handle)
#endregion