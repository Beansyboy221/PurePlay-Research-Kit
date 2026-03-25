import blinker
import inspect

class OnInitMixin():
    '''A mixin that sends a signal when a non-abstract class is initialized.'''
    on_init = blinker.Signal()

    def __init__(self):
        if not inspect.isabstract(self):
            self.on_init.send(self)