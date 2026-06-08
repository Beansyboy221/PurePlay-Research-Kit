"""A program mode that polls for and saves input data to files."""

import beanapp

from . import config

mode = beanapp.ProgramMode(
    name="Collect",
    description="Polls for and saves input data to files.",
    config_class=config.CollectConfig,
)
