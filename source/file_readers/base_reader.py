import abc

from utilities.mixins import on_init_mixin

class BaseReader(abc.ABC, on_init_mixin.OnInitMixin):
    '''Base class is simply for enforcing mixins.'''
    @property
    @abc.abstractmethod
    def data_type(self) -> type:
        '''The data type to read from a file.'''
        raise NotImplementedError
    
    # Read methods should be defined as strategies in protocols