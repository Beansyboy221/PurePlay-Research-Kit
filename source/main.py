from modes import mode_selector
from globals import logger
from startup import (
    startup_helpers,
    startup_config
)

def main():
    try:
        app_config = startup_config.StartupConfig()
        logger.set_log_level(app_config.log_level)
        config_dict = startup_helpers.load_config_file(app_config.config_path)
        mode_selector.select_and_run_mode(config_dict, app_config.use_gui)
    except Exception as e:
        logger.exception(e)

if __name__ == '__main__':
    main()