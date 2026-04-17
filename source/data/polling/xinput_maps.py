from data.polling import bind_enums

TRIGGER_MAP: dict[bind_enums.GamepadTrigger, callable] = {
    bind_enums.GamepadTrigger.LEFT_TRIGGER:  lambda t: t[0],
    bind_enums.GamepadTrigger.RIGHT_TRIGGER: lambda t: t[1],
}
'''Mapping of trigger enums to their positions in the output of XInput.get_trigger_values'''

STICK_MAP: dict[bind_enums.GamepadStick, callable] = {
    bind_enums.GamepadStick.LEFT_STICK_X:  lambda s: s[0][0],
    bind_enums.GamepadStick.LEFT_STICK_Y:  lambda s: s[0][1],
    bind_enums.GamepadStick.RIGHT_STICK_X: lambda s: s[1][0],
    bind_enums.GamepadStick.RIGHT_STICK_Y: lambda s: s[1][1],
}
'''Mapping of stick enums to their positions in the output of XInput.get_thumb_values'''