from prompting import cli
from preprocessing import scalers

@cli.register(scalers.SupportedScaler, 'Please select a supported scaler.')
def prompt_for_supported_scaler(data_type: type, title: str, message: str):
    raise NotImplementedError