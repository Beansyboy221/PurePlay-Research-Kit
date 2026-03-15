import keyboard
import XInput
import mouse
import math

from source.globals.constants import xinput_maps
from source.globals.enums import binds
from source.utilities.data_utils import windows_raw_mouse

#region Poll Mouse
def poll_mouse(mouse_whitelist: list[binds.MouseBind]) -> list:
    '''Returns a list of state values for mouse buttons and movement in the whitelist.'''
    delta_x = delta_y = magnitude = None
    row = []
    for bind in mouse_whitelist:
        if bind in binds.MouseButton:
            row.append(int(mouse.is_pressed(bind)))
        elif bind in binds.MouseAnalog:
            if magnitude is None:
                with windows_raw_mouse.mouse_lock:
                    delta_x = windows_raw_mouse.mouse_deltas[0]
                    delta_y = windows_raw_mouse.mouse_deltas[1]
                    windows_raw_mouse.mouse_deltas[0] = 0
                    windows_raw_mouse.mouse_deltas[1] = 0
                magnitude = math.hypot(delta_x, delta_y) # PyThagorus :/
            if bind == binds.MouseAnalog.DIRECTION_X:
                row.append(delta_x / magnitude if magnitude > 0 else 0.0)
            elif bind == binds.MouseAnalog.DIRECTION_Y:
                row.append(delta_y / magnitude if magnitude > 0 else 0.0)
            elif bind == binds.MouseAnalog.VELOCITY:
                row.append(math.log1p(magnitude))
        else:
            raise ValueError(f'Unknown bind: {bind}')
    return row
#endregion

#region Poll Keyboard
def poll_keyboard(keyboard_whitelist: list[binds.KeyBind]) -> list:
    '''Returns a list of state values for all binds in the given whitelist.'''
    return [int(keyboard.is_pressed(key)) for key in keyboard_whitelist]
#endregion

#region Poll Gamepad
def poll_gamepad(gamepad_whitelist: list[binds.GamepadBind]) -> list:
    '''Returns a list of state values for all binds in the given whitelist.'''
    if not XInput.get_connected()[0]:
        return [0] * len(gamepad_whitelist)
    gamepad_state = XInput.get_state(0)
    button_values = trigger_values = thumb_values = None
    row = []
    for bind in gamepad_whitelist:
        if bind in binds.GamepadButton:
            if button_values is None:
                button_values = XInput.get_button_values(gamepad_state)
            row.append(int(button_values[bind]))
        elif bind in binds.GamepadTrigger:
            if trigger_values is None:
                trigger_values = XInput.get_trigger_values(gamepad_state)
            row.append(xinput_maps.TRIGGER_MAP[bind](trigger_values))
        elif bind in binds.GamepadStick:
            if thumb_values is None:
                thumb_values = XInput.get_thumb_values(gamepad_state)
            row.append(xinput_maps.STICK_MAP[bind](thumb_values))
        else:
            raise ValueError(f'Unknown bind: {bind}')
    return row
#endregion

#region Other
def is_pressed(bind: binds.DigitalBind) -> bool:
    '''Determines whether a one-dimensional bind is pressed.'''
    if not bind:
        raise ValueError('Bind cannot be empty.')
    try:
        return mouse.is_pressed(bind)
    except Exception:
        pass
    try:
        return keyboard.is_pressed(bind)
    except Exception:
        pass
    try:
        if not XInput.get_connected()[0]:
            raise Exception('Gamepad not connected.')
        gamepad_state = XInput.get_state(0)
        button_values = XInput.get_button_values(gamepad_state)
        if button_values.get(bind):
            return True
        trigger_values = XInput.get_trigger_values(gamepad_state)
        if bind in xinput_maps.TRIGGER_MAP and xinput_maps.TRIGGER_MAP[bind](trigger_values) > 0:
            return True
        return False
    except Exception:
        raise Exception('Bind failed to parse.')

def should_kill(
        kill_binds: list[binds.DigitalBind], 
        kill_bind_gate: binds.BindGate = binds.BindGate.ANY
    ) -> bool:
    '''Determines whether the program should be terminated based on kill binds.'''
    if not kill_binds:
        return False
    pressed_kill_binds = (is_pressed(bind) for bind in kill_binds)
    match kill_bind_gate:
        case binds.BindGate.ANY:
            return any(pressed_kill_binds)
        case binds.BindGate.ALL:
            return all(pressed_kill_binds)
        case binds.BindGate.NONE:
            raise ValueError('Kill binds do not support NONE gate.')
        case _:
            raise ValueError(f'Unknown bind gate: {kill_bind_gate}')

def poll_if_capturing(
        capture_binds: list[binds.DigitalBind], 
        capture_bind_gate: binds.BindGate = binds.BindGate.ANY,
        keyboard_whitelist: list[binds.KeyBind] = [],
        mouse_whitelist: list[binds.MouseBind] = [],
        gamepad_whitelist: list[binds.GamepadBind] = [],
        ignore_empty_polls: bool = True,
        reset_mouse_on_release: bool = True
    ) -> list:
    '''Polls input devices if capture bind(s) are pressed.'''
    if not keyboard_whitelist and not mouse_whitelist and not gamepad_whitelist:
        return None
    
    pressed = (is_pressed(bind) for bind in capture_binds)
    match capture_bind_gate:
        case binds.BindGate.ANY:
            capturing = any(pressed)
        case binds.BindGate.ALL:
            capturing = all(pressed)
        case binds.BindGate.NONE:
            capturing = not any(pressed)
        case _:
            raise ValueError(f'Unknown bind gate: {capture_bind_gate}')
    
    if capturing:
        row = poll_keyboard(keyboard_whitelist) + poll_mouse(mouse_whitelist) + poll_gamepad(gamepad_whitelist)
        if not ignore_empty_polls or any(row):
            return row
    else:
        if reset_mouse_on_release:
            with windows_raw_mouse.mouse_lock:
                windows_raw_mouse.mouse_deltas[0] = 0
                windows_raw_mouse.mouse_deltas[1] = 0
    return None
#endregion