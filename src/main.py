import logging
import tomllib

from beanapp import StartupConfig


def main():
    logger = logging.getLogger()
    try:
        logger.info("Loading config from environment...")
        app_config = StartupConfig()
        logger.setLevel(app_config.log_level)

        logger.info(f"Loading config from file: {app_config.config_path}...")
        mode_config = tomllib.load(app_config.config_path)
        mode_config.update(app_config.model_dump())

        logger.info(f"Running mode: {app_config.mode.name}")
        app_config.mode.run(mode_config)
    except Exception as e:
        logger.exception(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()
