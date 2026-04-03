import pydantic

from . import global_logger

class StartupConfig(pydantic.BaseModel):
    config_path: pydantic.FilePath = pydantic.Field(
        description='The path to your app config file.'
    )
    '''The path to your app config file.'''

    use_gui: bool = pydantic.Field( # Replace this with a strategy?
        default=False,
        description='Whether to use the gui or cli for IO.'
    )
    '''Whether to use the gui or cli for IO.'''

    log_level: global_logger.LogLevel = pydantic.Field(
        default=global_logger.LogLevel.INFO,
        description='The maximum logging level of the application.'
    )
    '''The maximum logging level of the application.'''