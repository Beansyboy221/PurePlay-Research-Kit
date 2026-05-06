import pydantic
import typing
import torch

from misc import validators

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
    torch.optim.SparseAdam,
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
    # torch.optim.lr_scheduler.ChainedScheduler,
    torch.optim.lr_scheduler.ReduceLROnPlateau,
    torch.optim.lr_scheduler.CyclicLR,
    torch.optim.lr_scheduler.CosineAnnealingWarmRestarts,
    torch.optim.lr_scheduler.OneCycleLR,
    torch.optim.lr_scheduler.PolynomialLR,
)

SupportedOptimizer = typing.Annotated[
    typing.Type[torch.optim.Optimizer],
    pydantic.AfterValidator(
        lambda optimizer: validators.validate_in_collection(
            optimizer, SUPPORTED_OPTIMIZERS
        )
    ),
]

SupportedScheduler = typing.Annotated[
    typing.Type[torch.optim.lr_scheduler.LRScheduler],
    pydantic.AfterValidator(
        lambda scheduler: validators.validate_in_collection(
            scheduler, SUPPORTED_SCHEDULERS
        )
    ),
]
