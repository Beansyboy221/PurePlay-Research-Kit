import argparse
import logging
import constants, config, devices, logger

def main():
    args = _parse_args()
    app_config = config.load_app_config(
        config_path=args.config,
        mode_override=args.mode,
    )
    
    # Common Configs
    if app_config.mode != constants.AppMode.COLLECT:
        devices.optimize_cuda_for_hardware()
        logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
    
    match app_config.mode:
        case constants.AppMode.COLLECT:
            import modes.collect
            modes.collect.collect_input_data(app_config.config)
        case constants.AppMode.TRAIN:
            import modes.train
            modes.train.train_model(app_config.config)
        case constants.AppMode.TEST:
            import modes.test
            modes.test.run_static_analysis(app_config.config)
        case constants.AppMode.DEPLOY:
            import modes.deploy
            modes.deploy.run_live_analysis(app_config.config)
        case _:
            logger.error(f'Unsupported mode: {app_config.mode}')

def _parse_args():
    parser = argparse.ArgumentParser(description='PurePlay-Research-Kit')
    parser.add_argument(
        '-c',
        '--config',
        type=str,
        help='Path to TOML config file'
    )
    parser.add_argument(
        '-m',
        '--mode',
        type=str,
        choices=[mode.value for mode in constants.AppMode],
        help='Program mode'
    )
    return parser.parse_args()

if __name__ == '__main__':
    main()