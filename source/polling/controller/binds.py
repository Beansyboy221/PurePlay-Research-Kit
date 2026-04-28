import pygame._sdl2.controller

from polling import base_bind

class Bind(base_bind.Bind):
    def poll(self, controller: pygame._sdl2.controller.Controller):
        raise NotImplementedError

class Button(Bind):
    def poll(self, controller):
        return controller.get_button(self.id)

class Axis(Bind):
    def poll(self, controller):
        return controller.get_axis(self.id)

class Buttons:
    A           = Button(pygame.CONTROLLER_BUTTON_A)
    B           = Button(pygame.CONTROLLER_BUTTON_B)
    X           = Button(pygame.CONTROLLER_BUTTON_X)
    Y           = Button(pygame.CONTROLLER_BUTTON_Y)
    DPAD_UP     = Button(pygame.CONTROLLER_BUTTON_DPAD_UP)
    DPAD_DOWN   = Button(pygame.CONTROLLER_BUTTON_DPAD_DOWN)
    DPAD_LEFT   = Button(pygame.CONTROLLER_BUTTON_DPAD_LEFT)
    DPAD_RIGHT  = Button(pygame.CONTROLLER_BUTTON_DPAD_RIGHT)
    LB          = Button(pygame.CONTROLLER_BUTTON_LEFTSHOULDER)
    RB          = Button(pygame.CONTROLLER_BUTTON_RIGHTSHOULDER)
    LEFT_STICK  = Button(pygame.CONTROLLER_BUTTON_LEFTSTICK)
    RIGHT_STICK = Button(pygame.CONTROLLER_BUTTON_RIGHTSTICK)
    BACK        = Button(pygame.CONTROLLER_BUTTON_BACK)
    START       = Button(pygame.CONTROLLER_BUTTON_START)

class Triggers:
    LT = Axis(pygame.CONTROLLER_AXIS_TRIGGERLEFT)
    RT = Axis(pygame.CONTROLLER_AXIS_TRIGGERRIGHT)

class Sticks:
    LEFT_STICK_X  = Axis(pygame.CONTROLLER_AXIS_LEFTX)
    LEFT_STICK_Y  = Axis(pygame.CONTROLLER_AXIS_LEFTY)
    RIGHT_STICK_X = Axis(pygame.CONTROLLER_AXIS_RIGHTX)
    RIGHT_STICK_Y = Axis(pygame.CONTROLLER_AXIS_RIGHTY)

class Binds(Buttons, Triggers, Sticks):
    pass