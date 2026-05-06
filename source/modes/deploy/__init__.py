import modes
from . import config

mode = modes.ProgramMode(
    name="Deploy",
    description="Polls for and analyzes inputs in realtime.",
    config_class=config.DeployConfig,
)
