"""Prompters for types from the PurePlay models package."""

from beanapp import register
import beanml


@register(beanml.BaseModel)
def prompt_model(
    message: str = "Please choose a model.",
    data_type: type = beanml.BaseModel,
):
    options = "\n - ".join(beanml.AVAILABLE_MODELS.keys())
    while True:
        selection = input(f"\n{message}\nOptions:{options}\n")

        if model_class := beanml.AVAILABLE_MODELS.get(selection):
            return model_class()

        print("Invalid selection. Please try again.")
