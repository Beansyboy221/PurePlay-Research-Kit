from misc import logging_utils, dialogs
import modes
from startup import helpers, config


def main():
    logger = logging_utils.get_logger()
    try:
        app_config = config.StartupConfig()
        logger.setLevel(app_config.log_level)

        logger.info(f"Loading config from file: {app_config.config_path}...")
        config_dict = helpers.load_config_file(app_config.config_path)

        mode: modes.ProgramMode = app_config.mode_override
        if not mode:
            message = "Please select the mode you want to run."
            options = modes.AVAILABLE_MODES.keys()
            if app_config.use_gui:
                mode = dialogs.ask_select(message, options)
            else:
                mode = input(f"{message}\nOptions: {options}\n")
        logger.debug(f"Mode selected: {mode.name}")

        logger.info("Validating file config...")
        # Should I add the option to pass a mode from the command line?
        mode.config_class.model_validate(config_dict)

        logger.info("Running mode...")
        mode.run(config_dict, app_config.use_gui)
    except Exception:
        logger.exception("An unexpected error occurred.")


if __name__ == "__main__":
    main()
