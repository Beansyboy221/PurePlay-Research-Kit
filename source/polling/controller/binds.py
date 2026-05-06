import pygame._sdl2.controller

from polling import base_bind


class Bind(base_bind.Bind):
    def poll(self, controller: pygame._sdl2.controller.Controller):
        raise NotImplementedError


class Button(Bind):
    def poll(self, controller):
        return controller.get_button(self.poll_id)


class Axis(Bind):
    def poll(self, controller):
        return controller.get_axis(self.poll_id)


A = Button("A", pygame.CONTROLLER_BUTTON_A)
B = Button("B", pygame.CONTROLLER_BUTTON_B)
X = Button("X", pygame.CONTROLLER_BUTTON_X)
Y = Button("Y", pygame.CONTROLLER_BUTTON_Y)
DPAD_UP = Button("DPAD_UP", pygame.CONTROLLER_BUTTON_DPAD_UP)
DPAD_DOWN = Button("DPAD_DOWN", pygame.CONTROLLER_BUTTON_DPAD_DOWN)
DPAD_LEFT = Button("DPAD_LEFT", pygame.CONTROLLER_BUTTON_DPAD_LEFT)
DPAD_RIGHT = Button("DPAD_RIGHT", pygame.CONTROLLER_BUTTON_DPAD_RIGHT)
LB = Button("LB", pygame.CONTROLLER_BUTTON_LEFTSHOULDER)
RB = Button("RB", pygame.CONTROLLER_BUTTON_RIGHTSHOULDER)
LEFT_STICK = Button("LS", pygame.CONTROLLER_BUTTON_LEFTSTICK)
RIGHT_STICK = Button("RS", pygame.CONTROLLER_BUTTON_RIGHTSTICK)
BACK = Button("BACK", pygame.CONTROLLER_BUTTON_BACK)
START = Button("START", pygame.CONTROLLER_BUTTON_START)

LT = Axis("LT", pygame.CONTROLLER_AXIS_TRIGGERLEFT)
RT = Axis("RT", pygame.CONTROLLER_AXIS_TRIGGERRIGHT)

LEFT_STICK_X = Axis("LX", pygame.CONTROLLER_AXIS_LEFTX)
LEFT_STICK_Y = Axis("LY", pygame.CONTROLLER_AXIS_LEFTY)
RIGHT_STICK_X = Axis("RX", pygame.CONTROLLER_AXIS_RIGHTX)
RIGHT_STICK_Y = Axis("RY", pygame.CONTROLLER_AXIS_RIGHTY)
