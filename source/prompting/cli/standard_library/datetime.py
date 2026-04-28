import tkinter.simpledialog
import datetime

from prompting.cli import helpers

@helpers.register(datetime.date)
def prompt_for_date(data_type, root, title, message = 'Please enter a date/time (YYYY-MM-DD).'):
    return tkinter.simpledialog.askstring(title, message, parent=root)
