import typing

from misc import validated_types
from polling import base_bind, params
from polling.controller import helpers as controller_helpers, binds as controller_binds
from polling.keyboard import helpers as keyboard_helpers, binds as key_binds
from polling.mouse import helpers as mouse_helpers

# This file needs to be as optimized as possible.


def is_active(bind: base_bind.Bind) -> bool:
    """Determines whether a bind is active."""
    if isinstance(bind, controller_binds.Bind):
        return controller_helpers.is_active(bind)
    if isinstance(bind, key_binds.Bind):
        return keyboard_helpers.is_active(bind)
    else:
        return mouse_helpers.is_active(bind)


def are_active(
    binds: typing.Iterable[base_bind.Bind], gate: validated_types.GateCallable = any
) -> bool:
    """Determines whether a set of binds are active based on the provided gate."""
    return gate([is_active(bind) for bind in binds])


def poll_if_capturing(poll_params: params.PollParams) -> list | None:
    """Polls input devices if capture bind(s) are active."""
    if not any(poll_params.whitelist):
        return None

    capturing = are_active(poll_params.capture_binds, poll_params.capture_bind_gate)
    if not capturing:
        if poll_params.reset_mouse_on_release:
            mouse_helpers.mouse_move_listener.reset_deltas()
        return None

    poll = [
        *keyboard_helpers.poll_keyboard(poll_params.keyboard_whitelist),
        *mouse_helpers.poll_mouse(poll_params.mouse_whitelist),
        *controller_helpers.poll_controller(poll_params.controller_whitelist),
    ]
    if poll_params.ignore_empty_polls and not any(poll):
        return None

    return poll
