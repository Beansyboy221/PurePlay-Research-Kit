import tkinter.ttk
import enum

from prompting.cli import helpers

@helpers.register(enum.Enum, 'Please select an enum.')
def prompt_for_enum(data_type: type[enum.Enum], title: str, message: str):
    root = tkinter.Tk()
    root.title(title)
    tkinter.Label(master=root, text=message).pack()

    enum_value = tkinter.StringVar()
    for member in data_type:
        tkinter.ttk.Radiobutton(
            master=root, 
            text=member.name, 
            variable=enum_value, 
            value=member.name
        ).pack()
    
    tkinter.ttk.Button(
        master=root,
        text='Submit', 
        command=root.destroy()
    ).pack()
    root.mainloop()
    return data_type[enum_value.get()]