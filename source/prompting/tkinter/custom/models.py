import tkinter.ttk

from models import (
    helpers as model_helpers,
    base
)
from prompting.tkinter import helpers

@helpers.register(base.BaseModel)
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
        values=model_helpers.AVAILABLE_MODELS.keys(), 
        textvariable=model_name
    ).pack()
    
    tkinter.ttk.Button(
        master=root, 
        text='Submit', 
        command=root.destroy
    ).pack()
    root.mainloop()
    return model_helpers.AVAILABLE_MODELS[model_name.get()]