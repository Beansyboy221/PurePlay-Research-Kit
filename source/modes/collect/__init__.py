import modes
from . import config

mode = modes.ProgramMode(
    name="Collect",
    description="Polls for and saves input data to files.",
    config_class=config.CollectConfig,
)
