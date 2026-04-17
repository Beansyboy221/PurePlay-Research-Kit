import blinker
import inspect

class OnInitMixin():
    '''A mixin that sends a signal when a non-abstract class is initialized.'''
    on_init = blinker.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.on_init.send(self)

class OnInitSubclassMixin():
    '''A mixin that sends a signal when a non-abstract subclass is initialized.'''
    on_init_subclass = blinker.Signal()

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        if not inspect.isabstract(cls):
            cls.on_init_subclass.send(cls)