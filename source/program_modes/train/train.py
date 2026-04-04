import optuna.visualization
import logging
import time

from globals import formats
from utilities import cuda_helpers
from utilities.app_utils import global_logger
from utilities.data_utils import datamodule
from . import (
    objectives,
    callbacks,
    helpers,
    train_config
	)

def train(config: train_config.TrainConfig) -> None:
    '''
    Main entry point for training mode.
    Tunes and trains a model using optuna.
    '''
    cuda_helpers.optimize_cuda_for_hardware()
    logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
    kill_callback = callbacks.KillTrainingCallback(config)

    # Load and configure the data
    data_module = datamodule.PurePlayDataModule(
        data_params=config,
        batch_size=config.windows_per_batch,
        labeled_train_dirs={
            config.training_file_dir: int('benign'),
            config.cheat_training_file_dir: int('cheat')
        },
        labeled_validation_dirs= {
            config.validation_file_dir: int('benign'),
            config.cheat_validation_file_dir: int('cheat')
        }
    )
    if not config.windows_per_batch:
        helpers.tune_batch_size(config.model_class, data_module, kill_callback)
    
    # Start optuna study
    study = optuna.create_study(
        study_name=f'{config.model_class}-{time.strftime(formats.TIMESTAMP_FORMAT)}'
    )
    study.optimize(
        lambda trial: objectives.objective(
            trial=trial, 
            config=config,
            data_module=data_module,
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