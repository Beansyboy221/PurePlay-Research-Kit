import sklearn.preprocessing
import torch
import enum

#region Formats
TIMESTAMP_FORMAT = '%b-%d-%Y_%I-%M%p'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
#endregion

#region Bind Enums
KeyBind = enum.StrEnum('KeyBind', (
    'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 
    'n', 'o', 'p', 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z', 
    '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '+', '-', '*', 
    '/', '.', ',', '<', '>', '?', '!', '@', '#', '$', '%', '^', '&', 
    '(', ')', '_', '=', '{', '}', '[', ']', '|', '\\', ':', ';', 
    ' ', '~', 'enter', 'esc', 'backspace', 'tab', 'space', 'caps lock', 
    'num lock', 'scroll lock', 'home', 'end', 'page up', 'page down', 
    'insert', 'delete', 'left', 'right', 'up', 'down', 'f1', 'f2', 'f3', 
    'f4', 'f5', 'f6', 'f7', 'f8', 'f9', 'f10', 'f11', 'f12', 'print screen', 
    'pause', 'break', 'windows', 'menu', 'right alt', 'ctrl', 
    'left shift', 'right shift', 'left windows', 'left alt', 'right windows', 
    'alt gr', 'alt', 'shift', 'right ctrl', 'left ctrl'
))

MouseBind = enum.StrEnum('MouseBind', (
    'left', 'right', 'middle', 'x', 'x2', 'directionX', 'directionY', 'velocity'
))

GamepadBind = enum.StrEnum('GamepadBind', (
    'DPAD_UP', 'DPAD_DOWN', 'DPAD_LEFT', 'DPAD_RIGHT', 'START', 'BACK', 
    'LEFT_THUMB', 'RIGHT_THUMB', 'LEFT_SHOULDER', 'RIGHT_SHOULDER', 
    'A', 'B', 'X', 'Y', 'LT', 'RT', 'LX', 'LY', 'RX', 'RY'
))
#endregion

#region Bind Groups
MOUSE_ANALOGS = ('directionX', 'directionY', 'velocity')
#endregion

#region Tuning Params
MAX_HIDDEN_LAYERS = 4
MAX_HIDDEN_SIZE = 256

SUPPORTED_SCALERS = (
    #sklearn.preprocessing.StandardScaler,
    sklearn.preprocessing.RobustScaler,
    #sklearn.preprocessing.MinMaxScaler,
    #sklearn.preprocessing.MaxAbsScaler
)
SCALER_MAP = {optimizer.__name__: optimizer for optimizer in SUPPORTED_SCALERS}

SUPPORTED_OPTIMIZERS = (
    torch.optim.Adafactor,
    torch.optim.Adadelta,
    torch.optim.Adagrad,
    torch.optim.Adam,
    torch.optim.Adamax,
    torch.optim.AdamW,
    torch.optim.ASGD,
    torch.optim.LBFGS,
    torch.optim.NAdam,
    torch.optim.RAdam,
    torch.optim.RMSprop,
    torch.optim.Rprop,
    torch.optim.SGD,
    torch.optim.SparseAdam
)
OPTIMIZER_MAP = {optimizer.__name__: optimizer for optimizer in SUPPORTED_OPTIMIZERS}
OPTIMIZERS_WITH_WEIGHT_DECAY = {'Adam', 'AdamW', 'Adadelta', 'Adagrad', 'ASGD', 'NAdam', 'RAdam', 'RMSprop', 'SGD'}
OPTIMIZERS_WITH_MOMENTUM = {'SGD', 'RMSprop'}

SUPPORTED_SCHEDULERS = (
    
    torch.optim.lr_scheduler.MultiplicativeLR,
    torch.optim.lr_scheduler.StepLR,
    torch.optim.lr_scheduler.MultiStepLR,
    torch.optim.lr_scheduler.ConstantLR,
    torch.optim.lr_scheduler.LinearLR,
    torch.optim.lr_scheduler.ExponentialLR,
    torch.optim.lr_scheduler.SequentialLR,
    torch.optim.lr_scheduler.CosineAnnealingLR,
    #torch.optim.lr_scheduler.ChainedScheduler,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
    torch.optim.lr_scheduler.CyclicLR,
    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    torch.optim.lr_scheduler.OneCycleLR,
    torch.optim.lr_scheduler.PolynomialLR 
)
SCHEDULER_MAP = {scheduler.__name__: scheduler for scheduler in SUPPORTED_SCHEDULERS}
#endregion

#region Config Enums
class AppMode(enum.StrEnum):
    COLLECT = 'collect'
    TRAIN = 'train'
    TEST = 'test'
    DEPLOY = 'deploy'

class InputGate(enum.StrEnum):
    ANY = 'ANY'
    ALL = 'ALL'

class WindowType(enum.StrEnum):
    SLIDING = 'sliding'
    TUMBLING = 'tumbling'

class TrainingType(enum.StrEnum): # Using StrEnum for readability in other formats
    SUPERVISED = 'supervised'
    UNSUPERVISED = 'unsupervised'
#endregion