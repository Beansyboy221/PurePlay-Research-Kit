import lightning
import logging

from source.utilities.app_utils import cuda_helpers
from utilities.app_utils import global_logger
from utilities.model_utils import model_utils
from utilities.data_utils import datamodule
from utilities.app_utils import (
    config_helpers
)
from . import config

def test(config: config.ModeConfig) -> None:
    '''
    Main entry point for test mode.
    Performs static analysis on selected data files using a pre-trained model.
    '''
    cuda_helpers.optimize_cuda_for_hardware()
    logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
    model = model_utils.load_model(config.model_file)
    data_module = datamodule.PurePlayDataModule(
        data_params=model.data_params,
        model_class=model.__class__,
        testing_dir=config.testing_file_dir
    )
    data_module.scaler_manager.load(model.scaler_name, model.scaler_params)
    trainer = lightning.Trainer(logger=False, enable_checkpointing=False)
    trainer.test(model=model, datamodule=data_module)
    # Graphing?
    # The output of the test method saves reconstruction history to a file.

if __name__ == '__main__':
    try:
        args = config_helpers.parse_args()
        config_path, use_gui, log_level = config_helpers.get_global_configs(args)
        global_logger.set_log_level(log_level)
        config_dict = config_helpers.load_config_file(config_path)
        config_dict = config_helpers.populate_missing_fields(config.ModeConfig, config_dict, use_gui)
        config_object = config.ModeConfig.model_validate(config_dict)
        test(config_object)
    except Exception as e:
        global_logger.exception(e)