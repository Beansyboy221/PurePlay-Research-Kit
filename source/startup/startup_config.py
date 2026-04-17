import pydantic_settings
import pydantic

from prompting.cli import cli_prompter
from globals import logger

class StartupConfig(pydantic_settings.BaseSettings, cli_parse_args=True):
    config_path: pydantic.FilePath = pydantic.Field(
        default=cli_prompter.prompt_with_cli(
            pydantic.FilePath, 
            title='Config File Path',
            message='The path to your app config file.'
        )
    )
    '''The path to your app config file.'''

    use_gui: bool = pydantic.Field(default=False)
    '''Whether to use the gui or cli for IO.'''

    log_level: logger.LogLevel = pydantic.Field(
        default=logger.LogLevel.INFO
	)
    '''The logging level of the global app logger.'''