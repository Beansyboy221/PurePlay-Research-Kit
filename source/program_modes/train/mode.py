from source.ml_models import modelparams
from . import (
    config,
    train
)

ENTRY_POINT = train
CONFIG_CLASSES = (config.ModeConfig, modelparams.ModelParams)