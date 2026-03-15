import tkinter.filedialog
import tkinter.ttk
import pathlib

# PurePlay imports
from globals.constants import formats
from source.globals import global_logger

#region CLI Input
def get_input_from_cli(
        title: str,
        description: str = 'Please select an input option...',
        data_type: type = str,
        options: list = None,
        is_file: bool = False,
        is_dir: bool = False,
        file_types: list[tuple[str]] = [('All files', '*.*')],
        can_be_none: bool = False
    ):
    '''Prompts the user for input via the command line.'''
    prompt = f'\nMissing: {title}\n'
    if description:
        prompt += description

    if data_type == bool:
        options = formats.BOOL_STRINGS.keys()
    if options:
        prompt += f'\nOptions:{"\n - ".join(str(option) for option in options)}'

    if is_file:
        if data_type != str:
            raise TypeError('Tried to generate CLI for file-path of non-string type.')
        path_prompt = '\nEnter file path...'
        if file_types:
            path_prompt += f'Supported types:{"\n - ".join(
                f'{file_type}: {extension}'
                for file_type, extension in file_types
            )}'
        prompt += f'\n{path_prompt}\n'
    elif is_dir:
        prompt += '\nEnter directory path...\n'

    while True:
        input_text = input(prompt).strip()
        if not input_text:
            if can_be_none:
                return None
            global_logger.warning('Input cannot be empty. Please try again.')
            continue

        try:
            string_options = list(map(str, options))
            if options and input_text not in string_options:
                global_logger.warning(f'Invalid option. Choose from: {", ".join(string_options)}')
                continue

            if is_file or is_dir:
                path = pathlib.Path(input_text)
                if is_file and not path.is_file():
                    global_logger.warning('Path is not a valid file. Please try again.')
                    continue
                if is_dir and not path.is_dir():
                    global_logger.warning('Path is not a valid directory. Please try again.')
                    continue
                return path

            if data_type is bool:
                bool_option = formats.BOOL_STRINGS.get(input_text.lower())
                if bool_option is not None:
                    return bool_option
                else:
                    global_logger.warning('Invalid bool option. Please try again.')
                    continue

            return data_type(input_text)

        except (ValueError, TypeError):
            global_logger.warning(f'Invalid input. Expected type: {data_type.__name__}. Please try again.')
#endregion

#region GUI Input
def get_input_from_gui(
        title: str,
        description: str = 'Please select an input option.',
        data_type: type = str,
        options: list = None,
        is_file: bool = False,
        is_dir: bool = False,
        file_types: list[tuple] = [('All files', '*.*')],
        can_be_none: bool = False
    ):
    '''
    Generates a popup window to input a value.
    Supports text, numeric entry, categorical dropdowns, and file/directory browsing.
    '''
    result = []
    root = tkinter.Tk()
    root.title(title)
    root.attributes('-topmost', True)
    root.focus_force()

    frame = tkinter.ttk.Frame(root, padding='20 20 20 20')
    frame.pack(fill='both', expand=True)

    tkinter.ttk.Label(
        master=frame, 
        text=description, 
        font=('Helvetica', 10)
    ).pack(pady=(0, 10))

    # 1. Path (File or Directory)
    if is_file or is_dir:
        if data_type != str:
            global_logger.error('Tried to generate GUI for file-path of non-string type.')
            raise TypeError('Tried to generate GUI for file-path of non-string type.')
        path_var = tkinter.StringVar()
        entry_frame = tkinter.ttk.Frame(frame)
        entry_frame.pack(fill='x')

        entry = tkinter.ttk.Entry(
            master=entry_frame, 
            textvariable=path_var
        )
        entry.pack(side='left', fill='x', expand=True, padx=(0, 5))

        def _browse():
            if is_dir:
                path = tkinter.filedialog.askdirectory(title=description)
            else:
                path = tkinter.filedialog.askopenfilename(
                    title=description, 
                    filetypes=file_types
                )
            if path:
                path_var.set(path)

        tkinter.ttk.Button(
            master=entry_frame, 
            text='Browse', 
            command=_browse
        ).pack(side='right')
        get_val_func = lambda: path_var.get()

    # 2. Categorical (Dropdown)
    elif options:
        combo = tkinter.ttk.Combobox(frame, values=options, state='readonly')
        combo.pack(fill='x', pady=5)
        if options:
            combo.current(0)
        get_val_func = lambda: combo.get()

    # 3. Boolean (Checkbox)
    elif data_type == bool:
        bool_var = tkinter.BooleanVar()
        check = tkinter.ttk.Checkbutton(frame, text='Enable', variable=bool_var)
        check.pack(pady=5)
        get_val_func = lambda: bool_var.get()

    # 4. Numeric or String (Entry box)
    else:
        entry = tkinter.ttk.Entry(frame)
        entry.pack(fill='x', pady=5)
        entry.focus_set()
        get_val_func = lambda: entry.get()

    def _on_submit(event=None):
        val = get_val_func()
        if not val:
            if can_be_none:
                result.append(None)
                root.destroy()
            else:
                global_logger.warning('Input cannot be empty. Please try again.')
            return
        try:
            result.append(data_type(val))
            root.destroy()
        except ValueError:
            global_logger.warning('Invalid input. Please try again.')

    def _on_cancel(event=None):
        if can_be_none:
            result.append(None)
        else:
            global_logger.warning('Required input not provided.')
        root.destroy()

    # Submit/Cancel Buttons
    button_frame = tkinter.ttk.Frame(frame)
    button_frame.pack(pady=(15, 0))
    tkinter.ttk.Button(
        master=button_frame, 
        text='OK', 
        command=_on_submit
    ).pack(side='left', padx=5)
    tkinter.ttk.Button(
        master=button_frame, 
        text='Cancel (None)', 
        command=_on_cancel
    ).pack(side='left', padx=5)

    root.bind('<Return>', _on_submit)
    root.mainloop()

    return result[0] if result else None
#endregion