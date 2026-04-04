import pydantic

from . import global_logger

class StartupConfig(pydantic.BaseModel):
    config_path: pydantic.FilePath
    '''The path to your app config file.'''

    use_gui: bool = pydantic.Field(default=False)
    '''Whether to use the gui or cli for IO.'''

    log_level: global_logger.LogLevel = pydantic.Field(
        default=global_logger.LogLevel.INFO
	)
    '''The maximum logging level of the application.'''