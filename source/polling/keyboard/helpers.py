import pynput
import typing

from misc import validated_types
from . import binds

# This file needs to be as optimized as possible.

active_keybinds: set[binds.Bind] = set()

keyboard_listener = pynput.keyboard.Listener(
    on_press=lambda key: active_keybinds.add(key),
    on_release=lambda key: active_keybinds.discard(key),
)
keyboard_listener.start()


def poll_keyboard(keyboard_whitelist: frozenset[binds.Bind]) -> tuple:
    """Returns a list of state values for all binds in the given whitelist."""
    return [bind in active_keybinds for bind in keyboard_whitelist]


def is_active(bind: binds.Bind) -> bool:
    """Determines whether a bind is active."""
    return bind in active_keybinds


def are_active(
    binds: typing.Iterable[binds.Bind], gate: validated_types.GateCallable = any
) -> bool:
    """Determines whether a set of binds are active based on the provided gate."""
    return gate([is_active(bind) for bind in binds])
