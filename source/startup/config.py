from tkinter.filedialog import askopenfilename
import pydantic_settings
import pydantic

from misc import logging_utils
import modes

CFG_MESSAGE = "Please enter the path to your config file."


class StartupConfig(pydantic_settings.BaseSettings, cli_parse_args=True):
    """Settings for app startup. Supports running main with clargs."""

    use_gui: bool = pydantic.Field(default=True)
    """Whether to use a GUI or CLI for info IO."""

    config_path: pydantic.FilePath = pydantic.Field(
        default=pydantic.FilePath(
            askopenfilename(title=CFG_MESSAGE) if use_gui else input(CFG_MESSAGE)
        )
    )
    """The path to your app config file."""

    log_level: logging_utils.LogLevel = pydantic.Field(
        default=logging_utils.LogLevel.INFO
    )
    """The logging level of the global app logger."""

    mode_override: modes.ProgramMode = pydantic.Field(default=None)
    """The mode that will be run instead of the one found in the config file."""
