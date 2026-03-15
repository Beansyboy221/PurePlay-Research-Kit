import optuna.visualization
import logging
import time

# PurePlay imports
from source.globals.constants import formats
from source.globals import global_logger
from source.utilities.model_utils import modelparams
from source.utilities.app_utils import (
    config_utils,
    cuda_utils
)
from source.utilities.data_utils import (
    datamodule,
    dataparams
)

# Mode-specific imports
from .config import ModeConfig
from . import callbacks, objectives, helpers

def train(config: ModeConfig, model_params: modelparams.ModelParams) -> None:
    '''
    Main entry point for training mode.
    Tunes and trains a model using optuna.
    '''
    cuda_utils.optimize_cuda_for_hardware()
    logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
    kill_callback = callbacks.KillTrainingCallback(config)

    # Load and configure the data
    data_module = datamodule.PurePlayDataModule(
        data_params=dataparams.DataParams(
            whitelist=config.keyboard_whitelist + config.mouse_whitelist + config.gamepad_whitelist,
            ignore_empty_polls=config.ignore_empty_polls,
            polls_per_sequence=config.polls_per_sequence
        ),
        batch_size=config.sequences_per_batch,
        labeled_train_dirs={
            config.training_file_dir: 'benign', # Currently only supports 2 classes
            config.cheat_training_file_dir: 'cheat'
        },
        labeled_validation_dirs= {
            config.validation_file_dir: 'benign',
            config.cheat_validation_file_dir: 'cheat'
        }
    )
    if not config.sequences_per_batch:
        helpers.tune_batch_size(config.model_class, data_module, kill_callback)
    
    # Start optuna study
    study = optuna.create_study(
        study_name=f'{config.model_class}-{time.strftime(formats.TIMESTAMP_FORMAT)}'
    )
    study.optimize(
        lambda trial: objectives.objective(
            trial=trial, 
            model_class=config.model_class, 
            data_module=data_module,
            manual_params=model_params,
            save_dir=config.save_dir, 
            kill_callback=kill_callback
        ),
        callbacks=[kill_callback],
        gc_after_trial=True,
        show_progress_bar=True # Testing this feature currently
    )
    if len(study.trials) == 0:
        global_logger.warning('No trials completed. Exiting early.')
        return
    
    # Generate study reports
    global_logger.info(f'\nBest trial: {study.best_trial.number}')
    global_logger.info(f'Best val loss: {study.best_value:.6f}')
    if len(study.trials) == 1:
        global_logger.warning('Not enough trials completed for a comparison. Skipping visualization.')
        return
    figure = optuna.visualization.plot_optimization_history(study)
    figure.write_html(f'{config.save_dir}/optimization_history.html')
    figure = optuna.visualization.plot_param_importances(study)
    figure.write_html(f'{config.save_dir}/param_importances.html')
    global_logger.info('Report graphs saved.')

if __name__ == '__main__':
    try:
        args = config_utils.parse_args()
        config_path, use_gui, log_level = config_utils.get_global_configs(args)
        global_logger.set_log_level(log_level)
        config_dict = config_utils.load_config_file(config_path)
        config_dict = config_utils.populate_missing_fields(ModeConfig, config_dict, use_gui)
        config_object = ModeConfig.model_validate(config_dict)
        train(config_object)
    except Exception as e:
        global_logger.exception(e)