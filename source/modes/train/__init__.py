import modes
from . import config

mode = modes.ProgramMode(
    name="Train",
    description="Tunes and trains a model on a set of files.",
    config_class=config.TrainConfig,
)
