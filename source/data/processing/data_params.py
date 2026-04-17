import pydantic

from data.polling import poll_params

class ProcessingParams(pydantic.BaseModel):
    '''Parameters for data processing.'''
    ignore_empty_polls: bool = pydantic.Field(default=True) # Shared between polling and processing params.
    '''Whether or not empty rows of features should be written to the data.'''

    polls_per_window: int = pydantic.Field(
        default=128,
        multiple_of=2,
        ge=8
	)
    '''The number of polls(rows) of the whitelisted features in each window.'''

    window_stride: int = pydantic.Field(
        default=polls_per_window // 2,
        ge=1,
        lt=polls_per_window
	)
    '''The number of polls(rows) to skip between windows.'''

class DataParams(poll_params.PollParams, ProcessingParams):
    '''All data parameters.'''
    @property
    def features_per_window(self) -> int:
        '''The number of features in each window (sequence of polls).'''
        return self.features_per_poll * self.polls_per_window