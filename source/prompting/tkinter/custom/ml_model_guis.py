import tkinter.ttk

from model_manager import (
    base_model
)
from source.models import model_registry

from ....utilities.prompt_utils.tkinter import tkinter_prompter

@tkinter_prompter.register(base_model.BaseModel)
def prompt_for_base_model(
        data_type: type,
        title: str,
        message: str = 'Please select a model.'
    ):
    root = tkinter.Tk()
    root.title(title)
    tkinter.ttk.Label(master=root, text=message).pack()

    model_name = tkinter.StringVar()
    tkinter.ttk.Combobox(
        master=root, 
        values=model_registry.AVAILABLE_MODELS.keys(), 
        textvariable=model_name
    ).pack()
    
    tkinter.ttk.Button(
        master=root, 
        text='Submit', 
        command=root.destroy
    ).pack()
    root.mainloop()
    return model_registry.AVAILABLE_MODELS[model_name.get()]