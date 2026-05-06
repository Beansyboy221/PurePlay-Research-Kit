"""A framework for building ML models for PurePlay"""

import importlib
import pkgutil
import torch

from globals import processors
from misc import validated_types, logging_utils
from models import base

logger = logging_utils.get_logger()

AVAILABLE_MODELS: dict[str, type[base.BaseModel]] = {}
"""
A registry of all loaded models.
Key: model class name
Value: model class reference
"""


def register_model(model_class: type[base.BaseModel]) -> None:
    AVAILABLE_MODELS[model_class.__name__] = model_class
    logger.info(f"Model registered: {model_class.__name__}")


def load_model(checkpoint_path: validated_types.CheckpointPath) -> base.BaseModel:
    """Static method to load the correct child class automatically."""
    checkpoint = torch.load(
        f=checkpoint_path, map_location=processors.TORCH_DEVICE_TYPE
    )
    class_name = checkpoint.get("hyper_parameters").get("model_class")
    if not class_name:
        raise ValueError("Model file is missing model class name.")
    model_class = AVAILABLE_MODELS.get(class_name)
    if not model_class:
        raise ValueError(f"Model class {class_name} is not available.")
    return model_class.load_from_checkpoint(checkpoint_path)


base.BaseModel.on_init_concrete_subclass.connect(register_model)

logger.info("Searching for models to register...")
for loader, module_name, is_pkg in pkgutil.walk_packages(__path__, __name__ + "."):
    if is_pkg:
        continue
    logger.info(f"Importing module: {module_name}")
    importlib.import_module(module_name)
