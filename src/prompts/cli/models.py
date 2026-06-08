"""Prompters for types from the PurePlay models package."""

from beanapp import register
import beanml


@register(beanml.BaseModel)
def prompt_model(data_type: type, message: str):
    return beanml.AVAILABLE_MODELS.get(
        input(f"\n{message}\nOptions: {beanml.AVAILABLE_MODELS.keys()}")
    )
