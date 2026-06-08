"""Params needed for preprocessing."""

import tkinter.simpledialog

import pydantic

from beanapp import prompt
from beaninput import config
from beaninput.controller import binds as controller_binds
from beaninput.keyboard import binds as key_binds
from beaninput.mouse import binds as mouse_binds

DEFAULT_POLLS_PER_WINDOW = 128


class ProcessingParams(pydantic.BaseModel):
    """Parameters for data preprocessing."""

    keyboard_whitelist: frozenset[key_binds.Bind] = pydantic.Field(
        default_factory=frozenset,
        description="A set of keyboard input features to process.",
    )
    """A set of keyboard input features to process."""

    mouse_whitelist: frozenset[mouse_binds.Bind] = pydantic.Field(
        default_factory=frozenset,
        description="A set of mouse input features to process.",
    )
    """A set of mouse input features to process."""

    controller_whitelist: frozenset[controller_binds.Bind] = pydantic.Field(
        default_factory=frozenset,
        description="A set of controller input features to process.",
    )
    """A set of controller input features to process."""

    ignore_empty_polls: bool = pydantic.Field(
        default=True,
        description="Whether or not empty rows of features should be used for training.",
    )
    """Whether or not empty rows of features should be used for training."""

    polls_per_window: int = pydantic.Field(
        default_factory=lambda: prompt(
            cli_message="Please enter a window size (even number).",
            dialog=tkinter.simpledialog.askinteger(
                title="Polls per Window",
                prompt="Please choose a window size (even number).",
                minvalue=8,
                initialvalue=DEFAULT_POLLS_PER_WINDOW,
            ),
            prompt_default=DEFAULT_POLLS_PER_WINDOW,
        ),
        multiple_of=2,
        ge=8,
    )
    """The number of polls(rows) of the whitelisted features in each window."""

    window_stride: int = pydantic.Field(
        default_factory=lambda: prompt(
            cli_message="Please enter a window stride.",
            dialog=tkinter.simpledialog.askinteger(
                title="Window Stride",
                prompt="Please choose a window stride.",
                minvalue=1,
                maxvalue=DEFAULT_POLLS_PER_WINDOW - 1,
                initialvalue=DEFAULT_POLLS_PER_WINDOW // 2,
            ),
            prompt_default=DEFAULT_POLLS_PER_WINDOW // 2,
        ),
        multiple_of=2,
        ge=1,
        lt=DEFAULT_POLLS_PER_WINDOW,
    )
    """The number of polls(rows) to skip between windows."""


class DataParams(config.PollParams, ProcessingParams):
    """All data parameters. (For saving and loading datasets.)"""

    @property
    def features_per_window(self) -> int:
        """The number of features in each window (sequence of polls)."""
        return self.features_per_poll * self.polls_per_window
