import tomllib


def load_config_file(config_path: str) -> dict:
    """Loads a toml config file and returns a dictionary."""
    with open(config_path, "rb") as file_handle:
        return tomllib.load(file_handle)
