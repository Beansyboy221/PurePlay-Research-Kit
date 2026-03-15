from globals.enums import binds

#region XInput Mappings
'''Mapping of trigger enums to their positions in the output of XInput.get_trigger_values'''
TRIGGER_MAP = {
    binds.GamepadTrigger.LEFT_TRIGGER:  lambda t: t[0],
    binds.GamepadTrigger.RIGHT_TRIGGER: lambda t: t[1],
}

'''Mapping of stick enums to their positions in the output of XInput.get_thumb_values'''
STICK_MAP = {
    binds.GamepadStick.LEFT_STICK_X:  lambda s: s[0][0],
    binds.GamepadStick.LEFT_STICK_Y:  lambda s: s[0][1],
    binds.GamepadStick.RIGHT_STICK_X: lambda s: s[1][0],
    binds.GamepadStick.RIGHT_STICK_Y: lambda s: s[1][1],
}
#endregion