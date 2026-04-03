import abc

from utilities.mixins import on_init_mixin

class BaseWriter(abc.ABC, on_init_mixin.OnInitMixin):
    '''Base class is simply for enforcing mixins.'''
    @property
    @abc.abstractmethod
    def data_type(self) -> type:
        '''The data type to write to a file.'''
        raise NotImplementedError
    
    # Write methods should be defined as strategies in protocols