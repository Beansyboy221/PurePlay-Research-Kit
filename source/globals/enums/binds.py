import enum

class BindGate(enum.StrEnum):
    '''Enums for gated logic between binds.'''
    ANY = enum.auto()
    ALL = enum.auto()
    NONE = enum.auto()

#region Keyboard
class KeyboardButton(enum.StrEnum):
    '''Enums for keys recognized by the Keyboard module.'''
    # Letters
    A = 'a'; B = 'b'; C = 'c'; D = 'd'; E = 'e'; F = 'f'; G = 'g'
    H = 'h'; I = 'i'; J = 'j'; K = 'k'; L = 'l'; M = 'm'; N = 'n'
    O = 'o'; P = 'p'; Q = 'q'; R = 'r'; S = 's'; T = 't'; U = 'u'
    V = 'v'; W = 'w'; X = 'x'; Y = 'y'; Z = 'z'
    # Digits
    ZERO = '0'; ONE = '1'; TWO = '2'; THREE = '3'; FOUR = '4'
    FIVE = '5'; SIX = '6'; SEVEN = '7'; EIGHT = '8'; NINE = '9'
    # Symbols
    PLUS = '+'; MINUS = '-'; ASTERISK = '*'; SLASH = '/'; DOT = '.'
    COMMA = ','; LESS_THAN = '<'; GREATER_THAN = '>'; QUESTION = '?'; EXCLAIM = '!'
    AT = '@'; HASH = '#'; DOLLAR = '$'; PERCENT = '%'; CARET = '^'
    AMPERSAND = '&'; LPAREN = '('; RPAREN = ')'; UNDERSCORE = '_'
    EQUALS = '='; LBRACE = '{'; RBRACE = '}'; LBRACKET = '['; RBRACKET = ']'
    PIPE = '|'; BACKSLASH = '\\'; COLON = ':'; SEMICOLON = ';'
    SPACE = ' '; TILDE = '~'
    # Control keys
    ENTER = 'enter'; ESC = 'esc'; BACKSPACE = 'backspace'; TAB = 'tab'
    SPACE_KEY = 'space'; CAPS_LOCK = 'caps lock'; NUM_LOCK = 'num lock'
    SCROLL_LOCK = 'scroll lock'; HOME = 'home'; END = 'end'
    PAGE_UP = 'page up'; PAGE_DOWN = 'page down'
    INSERT = 'insert'; DELETE = 'delete'
    # Arrow keys
    LEFT_ARROW = 'left'; RIGHT_ARROW = 'right'
    UP_ARROW = 'up'; DOWN_ARROW = 'down'
    # Function keys
    F1 = 'f1'; F2 = 'f2'; F3 = 'f3'; F4 = 'f4'; F5 = 'f5'; F6 = 'f6'
    F7 = 'f7'; F8 = 'f8'; F9 = 'f9'; F10 = 'f10'; F11 = 'f11'; F12 = 'f12'
    # Modifier / special keys
    PRINT_SCREEN = 'print screen'; PAUSE = 'pause'; BREAK = 'break'
    WINDOWS = 'windows'; MENU = 'menu'
    CTRL = 'ctrl'; ALT = 'alt'; SHIFT = 'shift'
    LEFT_SHIFT = 'left shift'; RIGHT_SHIFT = 'right shift'
    LEFT_ALT = 'left alt'; RIGHT_ALT = 'right alt'
    LEFT_WINDOWS = 'left windows'; RIGHT_WINDOWS = 'right windows'
    LEFT_CTRL = 'left ctrl'; RIGHT_CTRL = 'right ctrl'
    ALT_GR = 'alt gr'
#endregion

#region Mouse
class MouseButton(enum.StrEnum):
    '''Enums for buttons recognized by the Mouse module.'''
    LEFT_MOUSE   = 'left'
    RIGHT_MOUSE  = 'right'
    MIDDLE_MOUSE = 'middle'
    MOUSE_4      = 'x'
    MOUSE_5      = 'x2'

class MouseAnalog(enum.StrEnum):
    '''Enums for mouse analogs (proprietary names).'''
    DIRECTION_X = enum.auto()
    DIRECTION_Y = enum.auto()
    VELOCITY    = enum.auto()
#endregion

#region XInput (Gamepad)
class GamepadButton(enum.StrEnum):
    '''Enums for buttons recognized by the XInput-Python module.'''
    A              = 'A'
    DPAD_UP        = 'DPAD_UP'
    DPAD_DOWN      = 'DPAD_DOWN'
    DPAD_LEFT      = 'DPAD_LEFT'
    DPAD_RIGHT     = 'DPAD_RIGHT'
    START          = 'START'
    BACK           = 'BACK'
    LEFT_THUMB     = 'LEFT_THUMB'
    RIGHT_THUMB    = 'RIGHT_THUMB'
    LEFT_SHOULDER  = 'LEFT_SHOULDER'
    RIGHT_SHOULDER = 'RIGHT_SHOULDER'
    GAMEPAD_A = 'A'; GAMEPAD_B = 'B' 
    GAMEPAD_X = 'X'; GAMEPAD_Y = 'Y'

class GamepadTrigger(enum.StrEnum):
    '''Enums for triggers recognized by the XInput-Python module.'''
    LEFT_TRIGGER = 'LT'
    RIGHT_TRIGGER = 'RT'

class GamepadStick(enum.StrEnum):
    '''Enums for sticks recognized by the XInput-Python module.'''
    LEFT_STICK_X = 'LX'; LEFT_STICK_Y = 'LY'
    RIGHT_STICK_X = 'RX'; RIGHT_STICK_Y = 'RY'
#endregion

#region Bind Groups
KeyBind = KeyboardButton
MouseBind = MouseButton | MouseAnalog
GamepadBind = GamepadButton | GamepadTrigger | GamepadStick
DigitalBind = KeyboardButton | MouseButton | GamepadButton | GamepadTrigger
Bind = KeyboardButton | MouseButton | MouseAnalog | GamepadButton | GamepadTrigger | GamepadStick
#endregion