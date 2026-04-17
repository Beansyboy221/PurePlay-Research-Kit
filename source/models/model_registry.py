import importlib
import pkgutil
import torch

from globals import processors
from models import base_model
from source.misc import validated_paths

AVAILABLE_MODELS: dict[str, type[base_model.BaseModel]] = {}
'''
A registry of all loaded models.
Use this to dynamically find models.
Key: model class name
Value: model class reference
'''

def register_model(model_class: type[base_model.BaseModel]) -> None:
    AVAILABLE_MODELS[model_class.__name__] = model_class
base_model.BaseModel.on_init_subclass.connect(register_model)

def load_model(checkpoint_path: validated_paths.CheckpointPath) -> base_model.BaseModel:
    '''Static method to load the correct child class automatically.'''
    checkpoint = torch.load(
        f=checkpoint_path, 
        map_location=processors.TORCH_DEVICE_TYPE
    )
    class_name = checkpoint.get('hyper_parameters').get('model_class')
    if not class_name:
        raise ValueError('Model file is missing model class name.')
    model_class = AVAILABLE_MODELS.get(class_name)
    if not model_class:
        raise ValueError(f'Model class {class_name} is not available.')
    return model_class.load_from_checkpoint(checkpoint_path)

for loader, module_name, is_pkg in pkgutil.walk_packages(__path__, __name__ + '.'):
    importlib.import_module(module_name)