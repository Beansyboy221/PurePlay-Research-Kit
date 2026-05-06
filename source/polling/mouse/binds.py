import pynput
import sys

from polling import base_bind


class Bind(base_bind.Bind):
    pass


class Button(Bind):
    pass


class Move(Bind):
    pass


MOUSE_LEFT = Button("LEFT_MOUSE", pynput.mouse.Button.left)
MOUSE_RIGHT = Button("RIGHT_MOUSE", pynput.mouse.Button.right)
MOUSE_MIDDLE = Button("MIDDLE_MOUSE", pynput.mouse.Button.middle)
if sys.platform == "win32":
    MOUSE_4 = Button("MOUSE_4", pynput.mouse.Button.x1)
    MOUSE_5 = Button("MOUSE_5", pynput.mouse.Button.x2)

DIRECTION_X = Move("MOUSE_X", "direction_x")
DIRECTION_Y = Move("MOUSE_Y", "direction_y")
VELOCITY = Move("MOUSE_VELOCITY", "velocity")
