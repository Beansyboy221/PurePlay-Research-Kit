import torch

from prompting.cli import helpers

@helpers.register(torch.optim.Optimizer, 'Please select an optimizer.')
def prompt_for_optimizer(data_type: type, title: str, message: str):
    raise NotImplementedError

@helpers.register(
    target_type=torch.optim.lr_scheduler.LRScheduler, 
    default_msg='Please select a learning rate scheduler.'
)
def prompt_for_lr_scheduler(data_type: type, title: str, message: str):
    raise NotImplementedError