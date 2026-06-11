"""Prompters for scalers supported by PurePlay."""

from beanapp import register

from preprocessing import scalers


@register(scalers.SupportedScaler)
def prompt_supported_scaler(
    message: str = "Please choose a supported scaler.",
    data_type: type = scalers.SupportedScaler,
):
    options = "\n - ".join(scalers.SUPPORTED_SCALERS.keys())
    while True:
        selection = input(f"\n{message}\nOptions:{options}\n")

        if scaler := scalers.SUPPORTED_SCALERS.get(selection):
            return scaler

        print("Invalid selection. Please try again.")
