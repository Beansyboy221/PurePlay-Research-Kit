"""A program mode that polls for and analyzes inputs in realtime."""

import beanapp

from . import config

mode = beanapp.ProgramMode(
    name="Deploy",
    description="Polls for and analyzes inputs in realtime.",
    config_class=config.DeployConfig,
)
