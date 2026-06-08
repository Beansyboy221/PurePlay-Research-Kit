import logging

import lightning

import beanml

from preprocessing import datamodule
from misc import cuda_helpers
from . import config


def main(config: config.TestConfig) -> None:
    """
    Main entry point for test mode.
    Performs static analysis on selected data files using a pre-trained model.
    """
    logger = logging.getLogger()
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

    logger.info("Configuring CUDA for your hardware...")
    cuda_helpers.optimize_cuda_for_hardware()

    logger.info(f"Loading model from file: {config.model_file}")
    model = beanml.load_model(config.model_file)
    data_module = datamodule.PurePlayDataModule(
        data_params=model.data_params, testing_dir=config.testing_file_dir
    )
    data_module.scaler = model.scaler

    logger.info(f"Analyzing files in dir: {config.testing_file_dir}")
    trainer = lightning.Trainer(logger=False, enable_checkpointing=False)
    trainer.test(model=model, datamodule=data_module)

    # Graphing?
    # The output of the test method currently
    # saves reconstruction history to a file.
