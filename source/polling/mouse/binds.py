import pynput
import sys

from polling import base_bind

class Bind(base_bind.Bind):
    pass

class Button(Bind):
    pass

class Move(Bind):
    pass

class Buttons:
    MOUSE_LEFT   = Button(pynput.mouse.Button.left)
    MOUSE_RIGHT  = Button(pynput.mouse.Button.right)
    MOUSE_MIDDLE = Button(pynput.mouse.Button.middle)
    if sys.platform == 'win32':
        MOUSE_4 = Button(pynput.mouse.Button.x1)
        MOUSE_5 = Button(pynput.mouse.Button.x2)

class Moves:
    DIRECTION_X = Move('direction_x')
    DIRECTION_Y = Move('direction_y')
    VELOCITY    = Move('velocity')

class Binds(Buttons, Moves):
    pass