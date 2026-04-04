import keyboard
import typing
import XInput
import mouse
import math

from utilities.data_utils import data_params
from . import (
    windows_raw_mouse,
    xinput_maps,
    bind_enums
	)

# This file needs to be as optimized as possible. (C++?)

#region Poll Mouse
def poll_mouse(mouse_whitelist: list[bind_enums.MouseBind]) -> list:
    '''Returns a list of state values for mouse buttons and movement in the whitelist.'''
    delta_x = delta_y = magnitude = None
    row = []
    for bind in mouse_whitelist:
        if isinstance(bind, bind_enums.MouseButton):
            row.append(mouse.is_pressed(bind))
        elif isinstance(bind, bind_enums.MouseAnalog):
            if magnitude is None:
                with windows_raw_mouse.mouse_lock:
                    delta_x = windows_raw_mouse.mouse_deltas[0]
                    delta_y = windows_raw_mouse.mouse_deltas[1]
                    windows_raw_mouse.mouse_deltas[0] = 0
                    windows_raw_mouse.mouse_deltas[1] = 0
                magnitude = math.hypot(delta_x, delta_y) # PyThagorus :/
            if bind == bind_enums.MouseAnalog.DIRECTION_X:
                row.append(delta_x / magnitude if magnitude > 0 else 0.0)
            elif bind == bind_enums.MouseAnalog.DIRECTION_Y:
                row.append(delta_y / magnitude if magnitude > 0 else 0.0)
            elif bind == bind_enums.MouseAnalog.VELOCITY:
                row.append(math.log1p(magnitude))
    return row
#endregion

#region Poll Keyboard
def poll_keyboard(keyboard_whitelist: list[bind_enums.KeyBind]) -> tuple:
    '''Returns a list of state values for all binds in the given whitelist.'''
    return (keyboard.is_pressed(key) for key in keyboard_whitelist)
#endregion

#region Poll Gamepad
def poll_gamepad(
        gamepad_whitelist: list[bind_enums.GamepadBind], 
        gamepad_index: int = 0
    ) -> list:
    '''Returns a list of state values for all binds in the given whitelist.'''
    if not XInput.get_connected()[gamepad_index]: # Move to separate thread
        return [0] * len(gamepad_whitelist)
    gamepad_state = XInput.get_state(gamepad_index)
    button_values = trigger_values = thumb_values = None
    row = []
    for bind in gamepad_whitelist:
        if isinstance(bind, bind_enums.GamepadButton):
            if button_values is None:
                button_values = XInput.get_button_values(gamepad_state)
            row.append(button_values[bind])
        elif isinstance(bind, bind_enums.GamepadTrigger):
            if trigger_values is None:
                trigger_values = XInput.get_trigger_values(gamepad_state)
            row.append(xinput_maps.TRIGGER_MAP[bind](trigger_values))
        elif isinstance(bind, bind_enums.GamepadStick):
            if thumb_values is None:
                thumb_values = XInput.get_thumb_values(gamepad_state)
            row.append(xinput_maps.STICK_MAP[bind](thumb_values))
    return row
#endregion

#region Other
def is_pressed(bind: bind_enums.DigitalBind, gamepad_index: int = 0) -> bool:
    '''Determines whether a one-dimensional bind is pressed.'''
    if isinstance(bind, bind_enums.MouseButton):
        return mouse.is_pressed(bind)
    if isinstance(bind, bind_enums.KeyBind):
        return keyboard.is_pressed(bind)
    if isinstance(bind, bind_enums.DigitalGamepadBind):
        if not XInput.get_connected()[gamepad_index]: # Move to separate thread
            raise RuntimeError('Gamepad not connected.')
        gamepad_state = XInput.get_state(gamepad_index)
        button_values = XInput.get_button_values(gamepad_state)
        if button_values.get(bind):
            return True
        trigger_values = XInput.get_trigger_values(gamepad_state)
        return xinput_maps.TRIGGER_MAP[bind](trigger_values) > 0

def are_pressed(
        binds: typing.Iterable[bind_enums.DigitalBind], 
        gate: bind_enums.BindGate = bind_enums.BindGate.ANY
    ) -> bool:
    '''Returns a bool after passing a list of binds through a gate.'''
    if len(binds) <= 1:
        raise ValueError('are_pressed() requires more than one bind.')
    pressed_binds = (is_pressed(bind) for bind in binds)
    match gate:
        case bind_enums.BindGate.ANY: # Is bindgate necessary?
            return any(pressed_binds)
        case bind_enums.BindGate.ALL:
            return all(pressed_binds)
        case bind_enums.BindGate.NONE:
            return not any(pressed_binds)
        
def poll_if_capturing(
        capture_binds: list[bind_enums.DigitalBind],
        capture_bind_gate: bind_enums.BindGate,
        data_params: data_params.ResolvedDataParams
    ) -> list | None:
    '''Polls input devices if capture bind(s) are pressed.'''
    if not any(data_params.whitelist):
        return None

    capturing = are_pressed(capture_binds, capture_bind_gate)
    if not capturing:
        if data_params.reset_mouse_on_release:
            with windows_raw_mouse.mouse_lock:
                windows_raw_mouse.mouse_deltas[0] = 0
                windows_raw_mouse.mouse_deltas[1] = 0
        return None
    
    row = [
        *poll_keyboard(data_params.keyboard_whitelist),
        *poll_mouse(data_params.mouse_whitelist),
        *poll_gamepad(data_params.gamepad_whitelist)
    ]
    if data_params.ignore_empty_polls and not any(row):
        return None
    
    return row
#endregion