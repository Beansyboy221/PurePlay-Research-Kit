"""A program mode that analyzes a set of given files."""

import beanapp

from . import config

mode = beanapp.ProgramMode(
    name="Test",
    description="Analyzes a set of given files.",
    config_class=config.TestConfig,
)
