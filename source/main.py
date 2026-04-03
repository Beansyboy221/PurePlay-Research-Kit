if __name__ == '__main__':
    try:
        from program_modes import mode_selector
        from utilities.app_utils import (
            global_logger,
            startup_utils
        )
        args = startup_utils.parse_args()
        config_path, use_gui, log_level = startup_utils.get_global_configs(args)
        global_logger.set_log_level(log_level)
        config_dict = startup_utils.load_config_file(config_path)
        mode_selector.select_and_run_mode(config_dict, use_gui)
    except Exception as e:
        global_logger.exception(e)