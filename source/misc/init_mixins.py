import blinker
import inspect

class OnInitMixin():
    on_init = blinker.Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(self, *args, **kwargs)
        self.on_init.send(self)

class OnInitConcreteSubclassMixin():
    '''A mixin that sends a signal when a non-abstract subclass is defined.'''
    on_init_concrete_subclass = blinker.Signal()

    def __init_subclass__(cls, *args, **kwargs):
        super().__init_subclass__(*args, **kwargs)
        if not inspect.isabstract(cls):
            cls.on_init_concrete_subclass.send(cls)