import threading
import keyboard
import win32gui
import XInput
import ctypes
import torch
import mouse
import math
import os
import constants, logger

#region ctypes Structures for Raw Input
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

#region Raw Input Listener (mouse only)
mouse_deltas = [0, 0]  # [delta_x, delta_y]
mouse_lock = threading.Lock()
USER32_LIBRARY = ctypes.windll.user32
HWND_MESSAGE = -3
WM_INPUT = 0x00FF # maybe I should use win32con?
WM_CLOSE = 0x0010
WM_DESTROY = 0x0002

def listen_for_mouse_movement(kill_event: threading.Event) -> None:
    """Listens for raw mouse input and updates global mouse delta values."""
    instance_handle = win32gui.GetModuleHandle(None)
    class_name = 'RawInputWindow'
    window = win32gui.WNDCLASS()
    window.hInstance = instance_handle
    window.lpszClassName = class_name
    window.lpfnWndProc = _raw_input_window_procedure
    win32gui.RegisterClass(window)
    window_handle = win32gui.CreateWindow(
        class_name,                     # lpClassName
        "Raw Input Hidden Window",      # lpWindowName
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
        raise ctypes.WinError()

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
    """Window procedure for processing raw input messages. Updates mouse deltas on WM_INPUT and exits on WM_DESTROY."""
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

#region Device Polling Functions
def poll_keyboard(keyboard_whitelist: list) -> list:
    """Returns a list of state values for all binds in the given whitelist."""
    return [1 if keyboard.is_pressed(key) else 0 for key in keyboard_whitelist]

def poll_mouse(mouse_whitelist: list) -> list:
    """Returns a list of state values for mouse buttons and movement in the whitelist."""
    row = []
    for button in mouse_whitelist:
        if button in ['left', 'right', 'middle', 'x', 'x2']:
            row.append(1 if mouse.is_pressed(button) else 0)
    if any(bind in mouse_whitelist for bind in constants.MOUSE_ANALOGS):
        with mouse_lock:
            delta_x, delta_y = mouse_deltas[0], mouse_deltas[1]
            mouse_deltas[0], mouse_deltas[1] = 0, 0
        
        # Direction X and Y are normalized to better represent raw behavior
        magnitude = math.sqrt(delta_x**2 + delta_y**2) # PyThagorus
        if 'directionX' in mouse_whitelist:
            row.append(delta_x / magnitude if magnitude > 0 else 0.0)
        if 'directionY' in mouse_whitelist:
            row.append(delta_y / magnitude if magnitude > 0 else 0.0)

        # Velocity is log-scaled to prioritize small movements while still capturing large ones
        if 'velocity' in mouse_whitelist:
            row.append(math.log1p(magnitude))
    return row

def poll_gamepad(gamepad_whitelist: list) -> list:
    """Returns a list of state values for all binds in the given whitelist."""
    row = []
    if not XInput.get_connected()[0]: 
        return [0] * len(gamepad_whitelist)
    gamepad_state = XInput.get_state(0)
    button_values = XInput.get_button_values(gamepad_state)
    for bind in gamepad_whitelist:
        if bind in button_values:
            row.append(1 if button_values[bind] else 0)
        else:
            if bind == 'LT':
                trigger_values = XInput.get_trigger_values(gamepad_state)
                row.append(trigger_values[0])
            elif bind == 'RT':
                trigger_values = XInput.get_trigger_values(gamepad_state)
                row.append(trigger_values[1])
            elif bind in ['LX', 'LY', 'RX', 'RY']:
                left_thumb, right_thumb = XInput.get_thumb_values(gamepad_state)
                if bind == 'LX':
                    row.append(left_thumb[0])
                elif bind == 'LY':
                    row.append(left_thumb[1])
                elif bind == 'RX':
                    row.append(right_thumb[0])
                elif bind == 'RY':
                    row.append(right_thumb[1])
            else:
                row.append(0)
    return row

def poll_all_devices(config: dict) -> list:
    """Polls all devices based on the given configuration and returns a combined list of state values."""
    return poll_keyboard(config.keyboard_whitelist) + poll_mouse(config.mouse_whitelist) + poll_gamepad(config.gamepad_whitelist)

def is_pressed(bind: str) -> bool:
    """Determines whether a one-dimensional bind is pressed."""
    if not bind:
        return True
    try:
        if mouse.is_pressed(bind):
            return True
    except:
        pass
    try:
        if keyboard.is_pressed(bind):
            return True
    except:
        pass
    try:
        gamepad_state = XInput.get_state(0)
        button_values = XInput.get_button_values(gamepad_state)
        if button_values[bind]:
            return True
    except:
        pass
    try:
        gamepad_state = XInput.get_state(0)
        trigger_values = XInput.get_trigger_values(gamepad_state)
        if bind == 'LT' and trigger_values[0] > 0:
            return True
        elif bind == 'RT' and trigger_values[1] > 0:
            return True
    except:
        pass
    return False

def should_kill(config: dict) -> bool:
    """Determines whether the program should be terminated based on kill binds."""
    if not config.kill_bind_list:
        return False
    pressed_kill_binds = [is_pressed(bind) for bind in config.kill_bind_list]
    if config.kill_bind_logic == 'ANY':
        return any(pressed_kill_binds)
    else: # 'ALL'
        return all(pressed_kill_binds)

def poll_if_capturing(config: dict) -> list:
    """Polls input devices if capture bind(s) are pressed."""
    capturing = True
    if len(config.capture_bind_list) > 1:
        pressed_capture_binds = [is_pressed(bind) for bind in config.capture_bind_list]
        if config.capture_bind_logic == 'ANY':
            capturing = any(pressed_capture_binds)
        else:
            capturing = all(pressed_capture_binds)
    elif not is_pressed(config.capture_bind_list[0]):
        capturing = False

    if capturing:
        row = poll_all_devices(config)
        if config.ignore_empty_polls and not (row.count(0) == len(row)):
            return row
        elif not config.ignore_empty_polls:
            return row
    else:
        if config.reset_mouse_on_release: # Should this reset all analogs?
            with mouse_lock:
                mouse_deltas[0], mouse_deltas[1] = 0, 0
    return None
#endregion

#region Processor Optimizations
TORCH_DEVICE_TYPE = 'cuda' if torch.cuda.is_available() else 'cpu'
TORCH_DEVICE = torch.device(TORCH_DEVICE_TYPE)
CPU_WORKERS = max(os.cpu_count()//2, 2) if TORCH_DEVICE_TYPE == 'cuda' else os.cpu_count()//2

def optimize_cuda_for_hardware() -> None:
    """Applies CUDA optimizations based on the detected GPU architecture."""
    if not torch.cuda.is_available():
        logger.warning("CUDA not available. Running on CPU.")
        return {}

    device = torch.cuda.current_device()
    major, minor = torch.cuda.get_device_capability(device)
    device_name = torch.cuda.get_device_name(device)

    has_tensor_cores = major >= 7
    has_tf32 = major >= 8
    has_bf16 = major >= 8

    logger.info(f"CUDA device: {device_name} (sm_{major}{minor})")
    logger.info(f"Tensor Cores: {has_tensor_cores} | TF32/BF16: {has_tf32}")

    precision = "medium" if has_tf32 else "high"
    torch.set_float32_matmul_precision(precision)
    logger.info(f"float32 matmul precision set to '{precision}'.")

    if has_bf16:
        torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = True
        logger.info("BF16 reduced precision reduction enabled.")

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    logger.info("cuDNN benchmark enabled (best for fixed input sizes).")

    return None
#endregion