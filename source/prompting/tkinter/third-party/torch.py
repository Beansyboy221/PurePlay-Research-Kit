import torch

from prompting.tkinter import helpers

@helpers.register(torch.optim.Optimizer, 'Please select an optimizer.')
def prompt_for_optimizer(data_type: type, title: str, message: str):
    raise NotImplementedError

@helpers.register(torch.optim.lr_scheduler.LRScheduler, 'Please select a learning rate scheduler.')
def prompt_for_lr_scheduler(data_type: type, title: str, message: str):
    raise NotImplementedError