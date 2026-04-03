from utilities.poll_utils import (
    windows_raw_mouse,
    poll_helpers
)

from . import config

# Duplicated from collect mode
def try_poll(config: config.ModeConfig) -> list | None: # Should this be here? Should it be mode specific?
    '''Polls input devices if capture bind(s) are pressed.'''
    whitelists = [
        config.keyboard_whitelist, 
        config.mouse_whitelist, 
        config.gamepad_whitelist
    ]
    if not any(whitelists):
        return None

    capturing = poll_helpers.are_pressed(config.capture_binds, config.capture_bind_gate)
    if not capturing:
        if config.reset_mouse_on_release: # How to pass this from the model?
            with windows_raw_mouse.mouse_lock:
                windows_raw_mouse.mouse_deltas[0] = 0
                windows_raw_mouse.mouse_deltas[1] = 0
        return None
    
    row = [
        *poll_helpers.poll_keyboard(config.keyboard_whitelist),
        *poll_helpers.poll_mouse(config.mouse_whitelist),
        *poll_helpers.poll_gamepad(config.gamepad_whitelist)
    ]
    if config.ignore_empty_polls and not any(row):
        return None
    
    return row