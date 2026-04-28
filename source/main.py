from misc import logging_utils
import modes
from prompting.cli import helpers
import prompting.tkinter
from startup import (
    helpers,
    config
)

def main():
    logger = logging_utils.get_logger()
    try:
        app_config = config.StartupConfig()
        logger.setLevel(app_config.log_level)

        logger.info(f'Loading config from file: {app_config.config_path}')
        config_dict = helpers.load_config_file(app_config.config_path)

        mode_selection_kwargs = {
            'data_type': modes.ProgramMode,
            'title': 'Program Mode',
            'message': f'Please select the mode you want to run.\n \
                         Options: {modes.AVAILABLE_MODES.keys()}',
        }
        if app_config.use_gui:
            prompt_method = prompting.cli.create_prompt
        else:
            prompt_method = prompting.tkinter.create_prompt
        mode: modes.ProgramMode = prompt_method(**mode_selection_kwargs)
        config_dict = helpers.fill_missing_fields(
            config_class=mode.config_class, 
            input_configs=config_dict, 
            prompt_method=prompt_method
        )
        mode.run(config_dict, app_config.use_gui) # Should I add the option to pass a mode and config directly from the command line?
    except Exception:
        logger.exception('An unexpected error occurred.')
 
if __name__ == '__main__':
    main()