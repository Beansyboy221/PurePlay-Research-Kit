import optuna.visualization
import logging
import time

from globals import formats
from misc import logging_utils, cuda_helpers
from preprocessing import datamodule
from . import config, objectives, callbacks, helpers


def main(config: config.TrainConfig) -> None:
    """
    Main entry point for training mode.
    Tunes and trains a model using optuna.
    """
    logger = logging_utils.get_logger()
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

    logger.info("Configuring CUDA for your hardware...")
    cuda_helpers.optimize_cuda_for_hardware()

    logger.info("Configuring DataModule...")
    data_module = datamodule.PurePlayDataModule(
        data_params=config,
        batch_size=config.windows_per_batch,
        labeled_train_dirs={
            config.training_file_dir: int("benign"),
            config.cheat_training_file_dir: int("cheat"),
        },
        labeled_validation_dirs={
            config.validation_file_dir: int("benign"),
            config.cheat_validation_file_dir: int("cheat"),
        },
    )

    kill_callback = callbacks.KillTrainingCallback(config)
    if not config.windows_per_batch:
        logger.info("Tuning batch size...")
        helpers.tune_batch_size(config.model_class, data_module, kill_callback)

    logger.info("Tuning/training model...")
    study = optuna.create_study(
        study_name=f"{config.model_class}-{time.strftime(formats.TIMESTAMP_FORMAT)}"
    )
    study.optimize(
        lambda trial: objectives.objective(
            trial=trial,
            config=config,
            data_module=data_module,
            kill_callback=kill_callback,
        ),
        callbacks=[kill_callback],
        gc_after_trial=True,
        show_progress_bar=True,  # Testing this feature currently
    )
    if len(study.trials) == 0:
        logger.warning("No trials completed. Exiting early.")
        return
    logger.info("Training complete.")

    # Reports
    logger.info(f"\nBest trial: {study.best_trial.number}")
    logger.info(f"Best val loss: {study.best_value:.6f}")
    if len(study.trials) == 1:
        logger.warning(
            "Not enough trials completed for a comparison. Skipping visualization."
        )
        return
    figure = optuna.visualization.plot_optimization_history(study)
    figure.write_html(f"{config.save_dir}/optimization_history.html")
    figure = optuna.visualization.plot_param_importances(study)
    figure.write_html(f"{config.save_dir}/param_importances.html")
    logger.info("Report graphs saved.")
