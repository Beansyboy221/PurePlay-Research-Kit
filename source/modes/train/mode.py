from models import modelparams

from .train import train
from .config import ModeConfig

ENTRY_POINT = train
CONFIG_CLASSES = (ModeConfig, modelparams.ModelParams)