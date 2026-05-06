import platform
import pynput
import typing
import math

from misc import validated_types
from . import binds

match platform.system():
    case "Windows":
        from polling.mouse.listeners import windows as raw_mouse
    case "Darwin":
        from polling.mouse.listeners import macos as raw_mouse
    case "Linux":
        from polling.mouse.listeners import linux as raw_mouse
    case _:
        raise NotImplementedError(f"Unsupported platform: {platform.system()}")

# This file needs to be as optimized as possible.


# Must be started for buttons to begin polling.
active_mouse_binds: set[binds.Button] = set()
mouse_button_listener = pynput.mouse.Listener(
    on_click=lambda button: active_mouse_binds.add(button),
    on_release=lambda button: active_mouse_binds.discard(button),
)  # How to add scrolling support?

# Must be started for moves to begin polling.
mouse_move_listener = raw_mouse.RawMouseListener()


def poll_mouse(mouse_whitelist: frozenset[binds.Bind]) -> list:
    """Returns a list of state values for mouse buttons and movement in the whitelist."""
    poll = [bind in active_mouse_binds for bind in mouse_whitelist]
    delta_x, delta_y = mouse_move_listener.get_deltas()
    mouse_move_listener.reset_deltas()
    magnitude = math.hypot(delta_x, delta_y)
    if binds.DIRECTION_X in mouse_whitelist:
        poll.append(delta_x / magnitude if magnitude > 0 else 0.0)
    if binds.DIRECTION_Y in mouse_whitelist:
        poll.append(delta_y / magnitude if magnitude > 0 else 0.0)
    if binds.VELOCITY in mouse_whitelist:
        poll.append(math.log1p(magnitude))
    return poll


def is_active(bind: binds.Bind) -> bool:
    """Determines whether a bind is active."""
    return bind in active_mouse_binds


def are_active(
    binds: typing.Iterable[binds.Bind], gate: validated_types.GateCallable = any
) -> bool:
    """Determines whether a set of binds are active based on the provided gate."""
    return gate([is_active(bind) for bind in binds])
