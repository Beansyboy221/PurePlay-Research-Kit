import torch

from .....prompt_utils.tkinter import tkinter_prompter

@tkinter_prompter.register(torch.optim.Optimizer)
def prompt_for_optimizer(
        data_type: type, 
        title: str, 
        message: str = 'Please select an optimizer.'
    ):
    raise NotImplementedError

@tkinter_prompter.register(torch.optim.lr_scheduler.LRScheduler)
def prompt_for_lr_scheduler(
        data_type: type, 
        title: str, 
        message: str = 'Please select a learning rate scheduler.'
    ):
    raise NotImplementedError