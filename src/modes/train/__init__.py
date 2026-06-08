"""A program mode that tunes and trains a model on a set of files."""

import beanapp

from . import config

mode = beanapp.ProgramMode(
    name="Train",
    description="Tunes and trains a model on a set of files.",
    config_class=config.TrainConfig,
)
