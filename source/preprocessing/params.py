import pydantic

from polling import params
from polling.controller import binds as controller_binds
from polling.keyboard import binds as key_binds
from polling.mouse import binds as mouse_binds

DEFAULT_POLLS_PER_WINDOW = 128

class ProcessingParams(pydantic.BaseModel):
    '''Parameters for data preprocessing.'''
    keyboard_whitelist: frozenset[key_binds.Bind] = pydantic.Field(
        default=frozenset()
    )
    '''A set of keyboard input features to process.'''

    mouse_whitelist: frozenset[mouse_binds.Bind] = pydantic.Field(
        default=frozenset()
    )
    '''A set of mouse input features to process.'''

    controller_whitelist: frozenset[controller_binds.Bind] = pydantic.Field(
        default=frozenset()
    )
    '''A set of controller input features to process.'''

    ignore_empty_polls: bool = pydantic.Field(default=True)
    '''Whether or not empty rows of features should be written to the data.'''

    polls_per_window: int = pydantic.Field(
        default=DEFAULT_POLLS_PER_WINDOW,
        multiple_of=2,
        ge=8
	)
    '''The number of polls(rows) of the whitelisted features in each window.'''

    window_stride: int = pydantic.Field(
        default=DEFAULT_POLLS_PER_WINDOW // 2,
        ge=1,
        lt=DEFAULT_POLLS_PER_WINDOW
	)
    '''The number of polls(rows) to skip between windows.'''

class DataParams(params.PollParams, ProcessingParams):
    '''All data parameters. (For saving and loading datasets.)'''
    @property
    def features_per_window(self) -> int:
        '''The number of features in each window (sequence of polls).'''
        return self.features_per_poll * self.polls_per_window