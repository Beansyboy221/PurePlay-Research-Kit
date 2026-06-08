"""Prompters for scalers supported by PurePlay."""

from beanapp import register

from preprocessing import scalers


@register(scalers.SupportedScaler)
def prompt_supported_scaler(data_type: type, message: str):
    return scalers.SCALER_CACHE[input(message)]
