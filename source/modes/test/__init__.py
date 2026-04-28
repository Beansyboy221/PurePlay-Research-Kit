import modes
from . import config

mode = modes.ProgramMode(
    name='Test',
    description='Analyzes a set of given files.',
    config_class=config.TestConfig
)