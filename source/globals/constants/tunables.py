import sklearn.preprocessing
import torch

SUPPORTED_SCALERS = (
    #sklearn.preprocessing.StandardScaler,
    sklearn.preprocessing.RobustScaler,
    #sklearn.preprocessing.MinMaxScaler,
    #sklearn.preprocessing.MaxAbsScaler
)
SCALER_MAP = {
    scaler.__name__: scaler 
    for scaler in SUPPORTED_SCALERS
}

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
OPTIMIZER_MAP = {
    torch.optim.Optimizer.__name__: torch.optim.Optimizer
    for torch.optim.Optimizer in SUPPORTED_OPTIMIZERS
}
OPTIMIZERS_WITH_WEIGHT_DECAY = [
    torch.optim.Optimizer.__name__ 
    for torch.optim.Optimizer in SUPPORTED_OPTIMIZERS
    if hasattr(torch.optim.Optimizer, 'weight_decay')
]
OPTIMIZERS_WITH_MOMENTUM = [
    torch.optim.Optimizer.__name__ 
    for torch.optim.Optimizer in SUPPORTED_OPTIMIZERS
    if hasattr(torch.optim.Optimizer, 'momentum')
]

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
SCHEDULER_MAP = {
    torch.optim.lr_scheduler.LRScheduler.__name__: torch.optim.lr_scheduler.LRScheduler 
    for torch.optim.lr_scheduler.LRScheduler in SUPPORTED_SCHEDULERS
}