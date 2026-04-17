import tkinter.simpledialog
import datetime

from ....utilities.prompt_utils.cli import cli_prompter

@cli_prompter._build_gui.register(datetime.date)
def prompt_for_date(data_type, root, title, message = 'Please enter a date/time (YYYY-MM-DD).'):
    return tkinter.simpledialog.askstring(title, message, parent=root)
