import tkinter.simpledialog
import tkinter.filedialog
import pydantic

from source.globals import logger
from .....prompt_utils.tkinter import tkinter_prompter

@tkinter_prompter.register(pydantic.BaseModel)
def prompt_for_pydantic_model(
        data_type: type[pydantic.BaseModel], 
        title: str, 
        message: str = 'Please populate this pydantic model.'
    ):
    field_dict = {}
    for field_name, field_info in data_type.model_fields.items():
        while True:
            field = tkinter_prompter.prompt_with_tkinter(
                data_type=field_info.annotation, 
                title=field_name, 
                message=field_info.description
            )
            if field is None and field_info.is_required():
                logger.warning('Field is required. Please try again.')
                continue
            break
        field_dict[field_name] = field
    return data_type.model_validate(field_dict)

#region Constrained Numerics
@tkinter_prompter.register(pydantic.PositiveInt)
@tkinter_prompter.register(pydantic.NonNegativeInt)
@tkinter_prompter.register(pydantic.NegativeInt)
@tkinter_prompter.register(pydantic.NonPositiveInt)
def prompt_for_constrained_ints(
        data_type: type, 
        title: str, 
        message: str = 'Please enter an integer.'
    ):
    return pydantic.TypeAdapter(data_type).validate_python(
        tkinter.simpledialog.askinteger(title, message)
    )

@tkinter_prompter.register(pydantic.PositiveFloat)
@tkinter_prompter.register(pydantic.NonNegativeFloat)
@tkinter_prompter.register(pydantic.NegativeFloat)
@tkinter_prompter.register(pydantic.NonPositiveFloat)
def prompt_for_constrained_floats(
        data_type: type, 
        title: str, 
        message: str = 'Please enter a number.'
    ):
    return pydantic.TypeAdapter(data_type).validate_python(
        tkinter.simpledialog.askfloat(title, message)
    )
#endregion

#region Dates/Times
@tkinter_prompter.register(pydantic.PastDate)
@tkinter_prompter.register(pydantic.FutureDate)
@tkinter_prompter.register(pydantic.PastDatetime)
@tkinter_prompter.register(pydantic.FutureDatetime)
@tkinter_prompter.register(pydantic.AwareDatetime)
@tkinter_prompter.register(pydantic.NaiveDatetime)
def prompt_for_dates(
        data_type: type, 
        title: str, 
        message: str = 'Please enter a date/time (YYYY-MM-DD).'
    ):
    return pydantic.TypeAdapter(data_type).validate_python(
        tkinter.simpledialog.askstring(title, message)
    )
#endregion

#region Paths
@tkinter_prompter.register(pydantic.FilePath)
def prompt_for_filepath(
        data_type: type, 
        title: str, 
        message: str = 'Please select a file path.'
    ):
    return pydantic.TypeAdapter(data_type).validate_python(
        tkinter.filedialog.askopenfilename(title=title, message=message)
    )

@tkinter_prompter.register(pydantic.DirectoryPath)
def prompt_for_directorypath(
        data_type: type, 
        title: str, 
        message: str = 'Please select a directory path.'
    ):
    return pydantic.TypeAdapter(data_type).validate_python(
        tkinter.filedialog.askdirectory(title=title, message=message)
    )

# Tkinter doesn't offer a filedialog for both files and directories.
@tkinter_prompter.register(pydantic.NewPath)
def prompt_for_newpath(
        data_type: type, 
        title: str, 
        message: str = 'Please select a location for a new file.'
    ):
    # asksaveasfilename is ideal for NewPath as it returns a path string without requiring the file to exist
    return pydantic.TypeAdapter(data_type).validate_python(
        tkinter.filedialog.asksaveasfilename(title=title, initialfile=message)
    )
#endregion

#region Network
@tkinter_prompter.register(pydantic.AnyUrl)
@tkinter_prompter.register(pydantic.AnyHttpUrl)
@tkinter_prompter.register(pydantic.FileUrl)
def prompt_for_url_types(
        data_type: type, 
        title: str, 
        message: str = 'Please enter the URL.'
    ):
    return pydantic.TypeAdapter(data_type).validate_python(
        tkinter.simpledialog.askstring(title, message)
    )

@tkinter_prompter.register(pydantic.IPvAnyAddress)
@tkinter_prompter.register(pydantic.IPvAnyInterface)
@tkinter_prompter.register(pydantic.IPvAnyNetwork)
def prompt_for_ip_types(
        data_type: type, 
        title: str, 
        message: str = 'Please enter an IP address, interface, or network.'
    ):
    return pydantic.TypeAdapter(data_type).validate_python(
        tkinter.simpledialog.askstring(title, message)
    )
#endregion

#region Specialized Strings
@tkinter_prompter.register(pydantic.EmailStr)
def prompt_for_email(
        data_type: type, 
        title: str, 
        message: str = 'Please enter an email address.'
    ):
    return pydantic.TypeAdapter(data_type).validate_python(
        tkinter.simpledialog.askstring(title, message)
    )

@tkinter_prompter.register(pydantic.SecretStr)
def prompt_for_secret(
        data_type: type, 
        title: str, 
        message: str = 'Please enter the sensitive value.'
    ):
    return pydantic.TypeAdapter(data_type).validate_python(
        tkinter.simpledialog.askstring(title, message)
    )

@tkinter_prompter.register(pydantic.Json)
def prompt_for_json(
        data_type: type, 
        title: str, 
        message: str = 'Please enter a valid JSON string.'
    ):
    return pydantic.TypeAdapter(data_type).validate_python(
        tkinter.simpledialog.askstring(title, message)
    )

@tkinter_prompter.register(pydantic.UUID4)
def prompt_for_uuid(
        data_type: type, 
        title: str, 
        message: str = 'Please enter a valid UUID.'
    ):
    return pydantic.TypeAdapter(data_type).validate_python(
        tkinter.simpledialog.askstring(title, message)
    )
#endregion