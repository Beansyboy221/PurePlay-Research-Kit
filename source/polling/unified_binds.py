from polling.controller import binds as controller_binds
from polling.keyboard import binds as key_binds
from polling.mouse import binds as mouse_binds

class Binds(controller_binds.Binds, key_binds.Binds, mouse_binds.Binds):
    pass