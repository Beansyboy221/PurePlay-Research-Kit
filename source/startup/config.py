import pydantic_settings
import pydantic

from misc import logging_utils
from prompting.cli import helpers

class StartupConfig(pydantic_settings.BaseSettings, cli_parse_args=True):
    config_path: pydantic.FilePath = pydantic.Field(
        default=helpers.create_prompt(
            pydantic.FilePath, 
            title='Config File Path',
            message='The path to your app config file.'
        )
    )
    '''The path to your app config file.'''

    use_gui: bool = pydantic.Field(default=False)
    '''Whether to use the gui or cli for IO.'''

    log_level: logging_utils.LogLevel = pydantic.Field(
        default=logging_utils.LogLevel.INFO
	)
    '''The logging level of the global app logger.'''