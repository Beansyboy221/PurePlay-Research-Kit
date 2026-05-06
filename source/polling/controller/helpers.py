import pygame._sdl2.controller
import typing

from misc import validated_types, logging_utils
from . import binds

# This file needs to be as optimized as possible.

logger = logging_utils.get_logger()
pygame._sdl2.controller.init()


def _update_controllers() -> None:
    """Checks for changes in connected controllers and updates the cache."""
    for index in range(pygame._sdl2.controller.get_count()):
        if pygame._sdl2.controller.is_controller(index):
            _controller_cache[index] = pygame._sdl2.controller.Controller(index)
            logger.info(
                f"Controller: {pygame._sdl2.controller.name_forindex(index)} cached at index: {index}"
            )


_controller_cache: dict[int, pygame._sdl2.controller.Controller] = {}
_update_controllers()


def _get_controller(index: int = 0) -> pygame._sdl2.controller.Controller:
    """Get or initialize a joystick instance for the given index."""
    if index not in _controller_cache:
        _update_controllers()
        if index not in _controller_cache:
            raise RuntimeError(f"No controller connected with index: {index}")
    return _controller_cache[index]


def poll_controller(
    controller_whitelist: frozenset[binds.Bind], index: int = 0
) -> list:
    """Returns a list of state values for all binds in the given whitelist."""
    controller = _get_controller(index)
    if not controller:
        return [0] * len(controller_whitelist)
    return [bind.poll(controller) for bind in controller_whitelist]


def is_active(bind: binds.Bind, index: int = 0) -> bool:
    """Determines whether a bind is active."""
    return bool(poll_controller([bind], index)[0])


def are_active(
    binds: typing.Iterable[binds.Bind],
    gate: validated_types.GateCallable = any,
    index: int = 0,
) -> bool:
    """Determines whether a set of binds are active based on the provided gate."""
    return gate([is_active(bind, index) for bind in binds])
