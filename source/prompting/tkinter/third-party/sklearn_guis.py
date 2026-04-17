import sklearn.base

from .....prompt_utils.tkinter import tkinter_prompter

@tkinter_prompter.register(sklearn.base.TransformerMixin)
def prompt_for_transformer(
        data_type: type, 
        title: str, 
        message: str = 'Please select a transformer.'
    ):
    raise NotImplementedError