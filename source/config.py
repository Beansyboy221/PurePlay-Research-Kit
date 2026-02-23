import tkinter.filedialog
import tkinter.ttk
import pydantic
import tkinter
import tomllib
import typing
import constants, models, logger

#region Custom Validators
def _validate_model_name(model_name: str) -> typing.Any:
    if model_name in models.AVAILABLE_MODELS:
        return models.AVAILABLE_MODELS[model_name]
    raise ValueError(f'Invalid model class name: {model_name}')

def _validate_even(value: int) -> int:
    if value % 2 != 0:
        raise ValueError(f'Value must be even, got {value}')
    return value
#endregion

#region App Mode Configs
class SharedConfig(pydantic.BaseModel):
    """Fields pulled directly from the root of the config file."""
    kill_bind_list: typing.List[str] = pydantic.Field(default_factory=lambda: ['ESC'])
    kill_bind_logic: typing.Literal['ANY', 'ALL'] = pydantic.Field(default='ANY')

class CollectConfig(SharedConfig):
    """Fields pulled from the [collect] section of the config file."""
    save_dir: pydantic.DirectoryPath = pydantic.Field(
        default='data',
        validation_alias=pydantic.AliasPath('collect', 'save_dir')
    )
    polling_rate: int = pydantic.Field(
        default=60, 
        validation_alias=pydantic.AliasPath('collect', 'polling_rate')
    )
    ignore_empty_polls: bool = pydantic.Field(
        default=True, 
        validation_alias=pydantic.AliasPath('collect', 'ignore_empty_polls')
    )
    reset_mouse_on_release: bool = pydantic.Field(
        default=True,
        validation_alias=pydantic.AliasPath('collect', 'reset_mouse_on_release')
    )
    capture_bind_list: typing.List[str] = pydantic.Field(
        default_factory=lambda: ['right'], 
        validation_alias=pydantic.AliasPath('collect', 'capture_bind_list')
    )
    capture_bind_logic: typing.Literal['ANY', 'ALL'] = pydantic.Field(
        default='ANY', 
        validation_alias=pydantic.AliasPath('collect', 'capture_bind_logic')
    )
    keyboard_whitelist: list[str] = pydantic.Field(
        default_factory=lambda: [],
        validation_alias=pydantic.AliasPath('collect', 'keyboard_whitelist')
    )
    mouse_whitelist: list[str] = pydantic.Field(
        default_factory=lambda: ['deltaX', 'deltaY'],
        validation_alias=pydantic.AliasPath('collect', 'mouse_whitelist')
    )
    gamepad_whitelist: list[str] = pydantic.Field(
        default_factory=lambda: [],
        validation_alias=pydantic.AliasPath('collect', 'gamepad_whitelist')
    )

class TrainConfig(SharedConfig):
    """Fields pulled from the [train] section of the config file."""
    save_dir: pydantic.DirectoryPath = pydantic.Field(
        default='models',
        validation_alias=pydantic.AliasPath('train', 'save_dir')
    )
    model_class: typing.Annotated[typing.Any, pydantic.BeforeValidator(_validate_model_name)] = pydantic.Field(
        validation_alias=pydantic.AliasPath('train', 'model_class')
    )
    ignore_empty_polls: bool = pydantic.Field(
        default=True, 
        validation_alias=pydantic.AliasPath('train', 'ignore_empty_polls')
    )
    polls_per_sequence: typing.Annotated[int, pydantic.AfterValidator(_validate_even)] = pydantic.Field(
        default=128, 
        validation_alias=pydantic.AliasPath('train', 'polls_per_sequence')
    )
    sequences_per_batch: typing.Annotated[int, pydantic.AfterValidator(_validate_even)] = pydantic.Field(
        default=32, 
        validation_alias=pydantic.AliasPath('train', 'sequences_per_batch')
    )
    training_file_dir: pydantic.DirectoryPath = pydantic.Field(
        validation_alias=pydantic.AliasPath('train', 'training_file_dir')
    )
    validation_file_dir: pydantic.DirectoryPath = pydantic.Field(
        validation_alias=pydantic.AliasPath('train', 'validation_file_dir')
    )
    cheat_training_file_dir: typing.Optional[pydantic.DirectoryPath] = pydantic.Field(
        validation_alias=pydantic.AliasPath('train', 'cheat_training_file_dir')
    )
    cheat_validation_file_dir: typing.Optional[pydantic.DirectoryPath] = pydantic.Field(
        validation_alias=pydantic.AliasPath('train', 'cheat_validation_file_dir')
    )
    keyboard_whitelist: list[str] = pydantic.Field(
        default_factory=lambda: [],
        validation_alias=pydantic.AliasPath('train', 'keyboard_whitelist')
    )
    mouse_whitelist: list[str] = pydantic.Field(
        default_factory=lambda: ['deltaX', 'deltaY'],
        validation_alias=pydantic.AliasPath('train', 'mouse_whitelist')
    )
    gamepad_whitelist: list[str] = pydantic.Field(
        default_factory=lambda: [],
        validation_alias=pydantic.AliasPath('train', 'gamepad_whitelist')
    )

class TestConfig(SharedConfig):
    """Fields pulled from the [test] section of the config file."""
    save_dir: pydantic.DirectoryPath = pydantic.Field(
        validation_alias=pydantic.AliasPath('test', 'save_dir')
    )
    model_file: pydantic.FilePath = pydantic.Field(
        validation_alias=pydantic.AliasPath('test', 'model_file')
    )
    testing_file_dir: pydantic.DirectoryPath = pydantic.Field(
        validation_alias=pydantic.AliasPath('test', 'testing_file_dir')
    )

class DeployConfig(SharedConfig):
    """Fields pulled from the [deploy] section of the config file."""
    save_dir: pydantic.DirectoryPath = pydantic.Field(
        validation_alias=pydantic.AliasPath('deploy', 'save_dir')
    )
    model_file: pydantic.FilePath = pydantic.Field(
        validation_alias=pydantic.AliasPath('deploy', 'model_file')
    )
    write_to_file: bool = pydantic.Field(
        default=True,
        validation_alias=pydantic.AliasPath('deploy', 'write_to_file')
    )
    deployment_window_type: constants.WindowType = pydantic.Field(
        validation_alias=pydantic.AliasPath('deploy', 'deployment_window_type')
    )
    capture_bind_list: typing.List[str] = pydantic.Field(
        default_factory=lambda: ['right'], 
        validation_alias=pydantic.AliasPath('deploy', 'capture_bind_list')
    )
    capture_bind_logic: typing.Literal['ANY', 'ALL'] = pydantic.Field(
        default='ANY', 
        validation_alias=pydantic.AliasPath('deploy', 'capture_bind_logic')
    )

class AppConfig(pydantic.BaseModel):
    mode: constants.AppMode
    config: CollectConfig | TrainConfig | TestConfig | DeployConfig

MODE_TO_MODEL: dict[constants.AppMode, type[pydantic.BaseModel]] = {
    constants.AppMode.COLLECT: CollectConfig,
    constants.AppMode.TRAIN: TrainConfig,
    constants.AppMode.TEST: TestConfig,
    constants.AppMode.DEPLOY: DeployConfig,
}
#endregion

#region Helpers
def _validate_config(
        config_dict: dict,
        mode: constants.AppMode,
    ) -> AppConfig:
    """Validates the config dict against the appropriate Pydantic model based on the mode."""
    try:
        config_class = MODE_TO_MODEL[mode]
        validated = config_class.model_validate(config_dict)
        logger.info(f'Config validated for mode: {mode}')
        return AppConfig(mode=mode, config=validated)
    except Exception as e:
        logger.error(e)
        raise

def _select_mode_with_gui() -> constants.AppMode:
    selected_mode = []
    root = tkinter.Tk()
    root.attributes('-topmost', True)
    root.focus_force()
    frame = tkinter.ttk.Frame(padding=5)
    frame.pack()
    tkinter.ttk.Label(frame, text='Select application mode:').pack()
    combo_box = tkinter.ttk.Combobox(frame, values=[mode.value for mode in constants.AppMode])
    combo_box.pack()
    def _save_and_exit() -> None:
        mode_str = combo_box.get()
        try:
            mode = constants.AppMode(mode_str)
            selected_mode.append(mode)
            root.destroy()
        except ValueError:
            logger.error(f'Invalid mode selected: {mode_str}')
    tkinter.ttk.Button(frame, text='OK', command=_save_and_exit).pack()
    root.mainloop()
    return selected_mode[0] if selected_mode else None

def _load_config_from_file(file_path: str) -> dict:
    """Loads a TOML config file from the specified path."""
    with open(file_path, 'rb') as file:
        return tomllib.load(file)

def _load_config_with_gui() -> dict:
    """Opens a file dialog to select a TOML config file and loads it."""
    root = tkinter.Tk()
    root.withdraw()
    root.attributes('-topmost', True)
    root.focus_force()
    file_path = tkinter.filedialog.askopenfilename(
        parent=root,
        title='Select TOML configuration file', 
        filetypes=[('TOML Files', '*.toml')]
    )
    root.destroy()
    
    if not file_path:
        raise ValueError('No configuration file selected.')
    with open(file_path, 'rb') as file:
        return tomllib.load(file)

def _get_model_class_from_gui() -> None:
    selected_model = []
    root = tkinter.Tk()
    root.attributes('-topmost', True)
    root.focus_force()
    frame = tkinter.ttk.Frame(padding=5)
    frame.pack()
    tkinter.ttk.Label(frame, text='Select model class:').pack()
    combo_box = tkinter.ttk.Combobox(frame, values=list(models.AVAILABLE_MODELS.keys()))
    combo_box.pack()
    def _save_and_exit() -> None:
        model = combo_box.get()
        if model in models.AVAILABLE_MODELS:
            selected_model.append(model)
            root.destroy()
        else:
            logger.error(f'Invalid model class: {model}')
    tkinter.ttk.Button(frame, text='OK', command=_save_and_exit).pack()
    root.mainloop()
    return selected_model[0] if selected_model else None

def _populate_missing_configs_with_gui(config: dict, mode: constants.AppMode) -> dict:
    config.setdefault('collect', {})
    config.setdefault('train', {})
    config.setdefault('test', {})
    config.setdefault('deploy', {})
    match mode:
        case constants.AppMode.COLLECT:
            collect_config = config['collect']
            if not collect_config.get('save_dir'):
                collect_config['save_dir'] = tkinter.filedialog.askdirectory(
                    title='Select data save directory'
                )
            pass
        case constants.AppMode.TRAIN:
            train_config = config['train']
            if not train_config.get('save_dir'):
                train_config['save_dir'] = tkinter.filedialog.askdirectory(
                    title='Select model save directory'
                )
            if not train_config.get('model_class'):
                train_config['model_class'] = _get_model_class_from_gui()
            if not train_config.get('training_file_dir'):
                train_config['training_file_dir'] = tkinter.filedialog.askdirectory(
                    title='Select training files directory'
                )
            if not train_config.get('validation_file_dir'):
                train_config['validation_file_dir'] = tkinter.filedialog.askdirectory(
                    title='Select validation files directory'
                )
            model_name = train_config.get('model_class')
            if model_name and model_name in models.AVAILABLE_MODELS:
                model_class = models.AVAILABLE_MODELS[model_name]
                if model_class.training_type == constants.TrainingType.SUPERVISED:
                    if not train_config.get('cheat_training_file_dir'):
                        train_config['cheat_training_file_dir'] = tkinter.filedialog.askdirectory(
                            title='Select cheat training files'
                        )
                    if not train_config.get('cheat_validation_file_dir'):
                        train_config['cheat_validation_file_dir'] = tkinter.filedialog.askdirectory(
                            title='Select cheat validation files'
                        )
        case constants.AppMode.TEST:
            test_config = config['test']
            if not test_config.get('save_dir'):
                test_config['save_dir'] = tkinter.filedialog.askdirectory(
                    title='Select report save directory'
                )
            if not test_config.get('model_file'):
                test_config['model_file'] = tkinter.filedialog.askopenfilename(
                    title='Select model file',
                    filetypes=[('Checkpoint Files', '*.ckpt')]
                )
            if not test_config.get('testing_file_dir'):
                test_config['testing_file_dir'] = tkinter.filedialog.askdirectory(
                    title='Select testing files directory'
                )
        case constants.AppMode.DEPLOY:
            deploy_config = config['deploy']
            if not deploy_config.get('save_dir'):
                deploy_config['save_dir'] = tkinter.filedialog.askdirectory(
                    title='Select data save directory'
                )
            if not deploy_config.get('model_file'):
                deploy_config['model_file'] = tkinter.filedialog.askopenfilename(
                    title='Select model file', 
                    filetypes=[('Checkpoint Files', '*.ckpt')]
                )
    return config
#endregion

#region Main Entry Point
def load_app_config(*, config_path: str | None, mode_override: constants.AppMode | None,) -> AppConfig:
    """Main entry point for loading and validating the application config."""
    mode = mode_override if mode_override is not None else _select_mode_with_gui()
    if config_path:
        config_dict = _load_config_from_file(config_path)
    else:
        config_dict = _load_config_with_gui()
        config_dict = _populate_missing_configs_with_gui(config_dict, mode)
    return _validate_config(config_dict, mode)
#endregion