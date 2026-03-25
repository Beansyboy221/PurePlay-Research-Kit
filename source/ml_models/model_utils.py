import torch

from . import base_model

AVAILABLE_MODELS: dict[str, type[base_model.BaseModel]] = {}
'''
A registry of all loaded models.
Use this to dynamically find models.
'''

def register_model(model_class: type[base_model.BaseModel]) -> None:
    AVAILABLE_MODELS[model_class.__name__] = model_class
base_model.BaseModel.on_init.connect(register_model)

def load_model(checkpoint_path: str, **kwargs) -> base_model.BaseModel:
    '''Static method to load the correct child class automatically.'''
    checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
    class_name = checkpoint.get('hyper_parameters').get('model_class')
    if not class_name:
        raise ValueError(f'Model file is missing model class name.')
    model_class = AVAILABLE_MODELS.get(class_name)
    if not model_class:
        raise ValueError(f'Model class {class_name} is not available.')
    return model_class.load_from_checkpoint(checkpoint_path, **kwargs)