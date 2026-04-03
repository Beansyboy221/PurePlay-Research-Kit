import pydantic

from utilities.poll_utils import (
    poll_params,
    bind_enums
)

class DataParams(
        pydantic.BaseModel,
        poll_params.PollParams
    ):
    '''Parameters that must be defined by the user.'''
    ignore_empty_polls: bool = pydantic.Field(
        default=True, 
        description='Whether or not empty rows of features should be written to the data.',
        validation_alias=pydantic.AliasPath('collect', 'ignore_empty_polls')
    )
    '''Whether or not empty rows of features should be written to the data.'''

    polls_per_window: int = pydantic.Field(
        default=128,
        multiple_of=2,
        ge=8, 
        description='The number of polls(rows) of the whitelisted features in each window.',
        validation_alias=pydantic.AliasPath('train', 'polls_per_window')
    )
    '''The number of polls(rows) of the whitelisted features in each window.'''

    window_stride: int = pydantic.Field(
        default=polls_per_window // 2,
        ge=1,
        lt=polls_per_window,
        description='The number of polls(rows) to skip between windows.',
        validation_alias=pydantic.AliasPath('train', 'window_stride')
    )
    '''The number of polls(rows) to skip between windows.'''

    @property
    def whitelist(self) -> list[bind_enums.Bind]:
        '''All device whitelists combined.'''
        return self.keyboard_whitelist + self.mouse_whitelist + self.gamepad_whitelist

    @property
    def features_per_poll(self) -> int:
        '''The number of features in each poll (row).'''
        return len(self.keyboard_whitelist + self.mouse_whitelist + self.gamepad_whitelist)

    @property
    def features_per_window(self) -> int:
        '''The number of features in each window (sequence of polls).'''
        return self.features_per_poll * self.polls_per_window

class ResolvedDataParams(DataParams):
    '''DataParams extended with properties sourced from the dataset itself.'''
    polling_rate: int = pydantic.Field(
        default=60, 
        gt=0,
        description='The polling rate used when polling the data (not the hardware polling rate).',
        validation_alias=pydantic.AliasPath('collect', 'polling_rate')
    )
    '''The polling rate used when polling the data (not the hardware polling rate).'''

    reset_mouse_on_release: bool = pydantic.Field(
        default=True,
        description='Whether or not mouse deltas are reset to 0 when the capture bind is released.',
        validation_alias=pydantic.AliasPath('collect', 'reset_mouse_on_release')
    )
    '''Whether or not mouse deltas are reset to 0 when the capture bind is released.'''