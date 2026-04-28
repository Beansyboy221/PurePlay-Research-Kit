from pynput.keyboard import Key, KeyCode
import pydantic

from polling import base_bind

class Bind(base_bind.Bind):
    @pydantic.field_validator('id', mode='before')
    @classmethod
    def validate_and_convert_id(cls, id: Key | KeyCode | str) -> Key | KeyCode:
        if isinstance(id, str):
            if len(id) == 1:
                return KeyCode.from_char(id)
            raise ValueError(f'Bind string must be a single character, got: "{id}"')
        return id

class Binds:
    # Letters
    A='a'; B='b'; C='c'; D='d'; E='e'; F='f'; G='g'; H='h'; I='i'; J='j'
    K='k'; L='l'; M='m'; N='n'; O='o'; P='p'; Q='q'; R='r'; S='s'; T='t'
    U='u'; V='v'; W='w'; X='x'; Y='y'; Z='z'

    # Digits
    ZERO='0'; ONE='1'; TWO='2'; THREE='3'; FOUR='4'
    FIVE='5'; SIX='6'; SEVEN='7'; EIGHT='8'; NINE='9'

    # Symbols
    PLUS='+'; MINUS='-'; ASTERISK='*'; SLASH='/'; DOT='.'
    COMMA=','; LESS_THAN='<'; GREATER_THAN='>'; QUESTION='?'
    EXCLAIM='!'; AT='@'; HASH='#'; DOLLAR='$'; PERCENT='%'
    CARET='^'; AMPERSAND='&'; LPAREN='('; RPAREN=')'; UNDERSCORE='_'
    EQUALS='='; LBRACE='{'; RBRACE='}'; LBRACKET='['; RBRACKET=']'
    PIPE='|'; BACKSLASH='\\'; COLON=':'; SEMICOLON=';'; TILDE='~'
    BACKTICK='`'; APOSTROPHE="'"; QUOTE='"'; SPACE_KEY=' '

    # Control keys
    ENTER=Key.enter; ESC=Key.esc; BACKSPACE=Key.backspace; TAB=Key.tab
    SPACE=Key.space; CAPS_LOCK=Key.caps_lock; NUM_LOCK=Key.num_lock
    HOME=Key.home; END=Key.end; PAGE_UP=Key.page_up; PAGE_DOWN=Key.page_down
    INSERT=Key.insert; DELETE=Key.delete

    # Arrow keys
    UP=Key.up; DOWN=Key.down; LEFT=Key.left; RIGHT=Key.right

    # Modifiers
    CTRL=Key.ctrl; ALT=Key.alt; SHIFT=Key.shift; CMD=Key.cmd
    CTRL_L=Key.ctrl_l; CTRL_R=Key.ctrl_r; ALT_L=Key.alt_l; ALT_R=Key.alt_r
    SHIFT_L=Key.shift_l; SHIFT_R=Key.shift_r; CMD_L=Key.cmd_l; CMD_R=Key.cmd_r
    
    # Function Keys
    F1=Key.f1; F2=Key.f2; F3=Key.f3; F4=Key.f4; F5=Key.f5
    F6=Key.f6; F7=Key.f7; F8=Key.f8; F9=Key.f9; F10=Key.f10
    F11=Key.f11; F12=Key.f12; F13=Key.f13; F14=Key.f14; F15=Key.f15
    F16=Key.f16; F17=Key.f17; F18=Key.f18; F19=Key.f19; F20=Key.f20

for key, value in list(Binds.__dict__.items()):
    if key.startswith('__') or key == '_digit_names':
        continue
    if isinstance(value, (str, Key, KeyCode)):
        setattr(Binds, key, Bind(id=value))