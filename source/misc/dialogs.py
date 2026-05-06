"""Custom tkinter dialogs with extended capabilities."""

import tkinter.ttk
import typing

T = typing.TypeVar("T")


class BaseDialog(tkinter.Toplevel):
    """A reusable base class for dialogs."""

    def __init__(self, title: str, msg: str):
        super().__init__()
        self.title(title)
        self.columnconfigure(0, weight=1)
        self.result = None

        tkinter.Label(self, text=msg, pady=10).grid(row=0, column=0, columnspan=2)
        self.content_frame = tkinter.Frame(self)
        self.content_frame.grid(
            row=1, column=0, columnspan=2, sticky="nsew", padx=10, pady=10
        )
        self.setup_content()
        tkinter.Button(self, text="Cancel", command=self.destroy).grid(
            row=2, column=0, sticky="ew"
        )
        tkinter.Button(self, text="Ok", command=self.submit).grid(
            row=2, column=1, sticky="ew"
        )

        self.wait_window()

    def setup_content(self):
        """Override this in subclasses."""
        pass

    def submit(self):
        """Override this to set self.result."""
        self.destroy()


class SelectionDialog(BaseDialog):
    def __init__(self, title, msg, options):
        self.options = list(options)
        super().__init__(title, msg)

    def setup_content(self):
        self.combo = tkinter.ttk.Combobox(
            self.content_frame, values=self.options, state="readonly"
        )
        self.combo.pack(fill="x", expand=True)
        if self.options:
            self.combo.current(0)

    def submit(self):
        self.result = self.combo.get()
        super().submit()


class ListSelectorDialog(BaseDialog):
    def __init__(self, title, msg, options):
        self.options = list(options)
        super().__init__(title, msg)

    def setup_content(self):
        self.list_var = tkinter.Variable(value=[])

        # Listbox and Scrollbar
        list_frame = tkinter.Frame(self.content_frame)
        list_frame.pack(fill="both", expand=True)

        self.listbox = tkinter.Listbox(
            list_frame, listvariable=self.list_var, selectmode="multiple"
        )
        scrollbar = tkinter.Scrollbar(list_frame, command=self.listbox.yview)
        self.listbox.configure(yscrollcommand=scrollbar.set)

        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

    def submit(self):
        self.result = self.list_var.get()
        super().submit()


class ListBuilderDialog(BaseDialog):
    def __init__(self, title, msg, add_func):
        self.add_func = add_func
        super().__init__(title, msg)

    def setup_content(self):
        self.list_var = tkinter.Variable(value=[])

        # Listbox and Scrollbar
        list_frame = tkinter.Frame(self.content_frame)
        list_frame.pack(fill="both", expand=True)

        self.listbox = tkinter.Listbox(list_frame, listvariable=self.list_var)
        scrollbar = tkinter.Scrollbar(
            list_frame, selectmode="extended", command=self.listbox.yview
        )
        self.listbox.configure(yscrollcommand=scrollbar.set)

        self.listbox.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Action Buttons
        btn_frame = tkinter.Frame(self.content_frame)
        btn_frame.pack(fill="x")
        tkinter.Button(btn_frame, text="Add", command=self.add_item).pack(
            side="left", fill="x", expand=True
        )
        tkinter.Button(btn_frame, text="Remove", command=self.remove_item).pack(
            side="left", fill="x", expand=True
        )

    def add_item(self):
        if item := self.add_func():
            current = list(self.list_var.get())
            current.append(item)
            self.list_var.set(current)

    def remove_item(self):
        indices = self.listbox.curselection()
        current = list(self.list_var.get())
        for i in reversed(indices):
            current.pop(i)
        self.list_var.set(current)

    def submit(self):
        self.result = list(self.list_var.get())
        super().submit()


# region API


def ask_select(options: typing.Iterable[T], msg="Select an option.") -> T | None:
    """Creates a single selection dialog and return the result."""
    dialog = SelectionDialog("Select an option...", msg, options)
    return dialog.result


def ask_select_list(
    options: typing.Iterable[T], msg="Select a list of options."
) -> list[T] | None:
    """Creates a list selection dialog and return the result."""
    dialog = ListSelectorDialog("Select your options...", msg, options)
    return dialog.result


def ask_build_list(
    add_func: typing.Callable[[], T], msg="Add items to your list."
) -> list[T] | None:
    """Creates a list building dialog and return the result."""
    dialog = ListBuilderDialog("Build your list...", msg, add_func)
    return dialog.result


if __name__ == "__main__":
    fav_fruit = ask_select(
        ["Apple", "Banana", "Orange"], "What is your favorite fruit?"
    )
    print(f"User chose: {fav_fruit}")
    my_items = ask_build_list(lambda: "New Item", "Create your shopping list")
    print(f"Final list: {my_items}")
