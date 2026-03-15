import lightning
import logging

# PurePlay imports
from source.globals import global_logger
from source.utilities.model_utils import basemodel
from source.utilities.data_utils import datamodule
from source.utilities.app_utils import (
    config_utils,
    cuda_utils
)

# Mode-specific imports
from .config import ModeConfig

def test(config: ModeConfig) -> None:
    '''
    Main entry point for test mode.
    Performs static analysis on selected data files using a pre-trained model.
    '''
    cuda_utils.optimize_cuda_for_hardware()
    logging.getLogger('lightning.pytorch').setLevel(logging.ERROR)
    model = basemodel.BaseModel.load_model(config.model_file)
    data_module = datamodule.PurePlayDataModule(
        data_params=model.data_params,
        model_class=model.__class__,
        testing_dir=config.testing_file_dir
    )
    data_module.load_scaler(model.scaler_name, model.scaler_params)
    trainer = lightning.Trainer(logger=False, enable_checkpointing=False)
    trainer.test(model=model, datamodule=data_module)
    # Graphing?
    # The output of the test method saves reconstruction history to a file.

if __name__ == '__main__':
    try:
        args = config_utils.parse_args()
        config_path, use_gui, log_level = config_utils.get_global_configs(args)
        global_logger.set_log_level(log_level)
        config_dict = config_utils.load_config_file(config_path)
        config_dict = config_utils.populate_missing_fields(ModeConfig, config_dict, use_gui)
        config_object = ModeConfig.model_validate(config_dict)
        test(config_object)
    except Exception as e:
        global_logger.exception(e)