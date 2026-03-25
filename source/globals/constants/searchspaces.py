import sklearn.preprocessing
import torch

# Pre-Training
BATCH_SIZE_TUNE_EPOCHS = 50

# Early Stopping
EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_DELTA = 0.0001

# Hyperparameters
HIDDEN_SIZE_MIN = 16
LATENT_SIZE_MIN = 8
HIDDEN_LAYERS_MIN = 1
HIDDEN_LAYERS_MAX = 4
DROPOUT_MIN = 0.0
DROPOUT_MAX = 0.5
LEARNING_RATE_MIN = 1e-6
LEARNING_RATE_MAX = 1e-3
WEIGHT_DECAY_MIN = 1e-6
WEIGHT_DECAY_MAX = 1e-3
MOMENTUM_MIN = 0.0
MOMENTUM_MAX = 0.99

# Stochastic Weight Averaging
SWA_EPOCH_START_MIN = 20
SWA_EPOCH_START_MAX = 2000
SWA_LR_FACTOR_MIN = 0.05
SWA_LR_FACTOR_MAX = 0.8

#region Algorithms
SUPPORTED_SCALERS: frozenset[type[sklearn.base.TransformerMixin]] = (
    sklearn.preprocessing.StandardScaler, # May want to consider having another version where with_mean=False.
    sklearn.preprocessing.MinMaxScaler,
    sklearn.preprocessing.MaxAbsScaler # I think this will be the best choice.
)

SUPPORTED_OPTIMIZERS: frozenset[type[torch.optim.Optimizer]] = (
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

SUPPORTED_SCHEDULERS: frozenset[type[torch.optim.lr_scheduler.LRScheduler]] = (
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
#endregion