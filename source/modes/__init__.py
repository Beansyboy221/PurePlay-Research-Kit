"""A dynamic program mode loading system."""

import importlib
import pydantic
import pkgutil
import os

from misc import logging_utils, init_mixins

logger = logging_utils.get_logger("ModeManager")


class ModeConfig(pydantic.BaseModel):
    model_config = pydantic.ConfigDict(
        populate_by_name=True, arbitrary_types_allowed=True
    )


class ProgramMode(pydantic.BaseModel, init_mixins.OnInitMixin):
    name: str
    description: str
    config_class: type[ModeConfig]

    def run(self, raw_config: dict):
        """Imports and runs the entry point."""
        logger.info(f"Running mode: {self.name}")
        logic_module_path = f"{self.__class__.__module__}.main"
        module = importlib.import_module(logic_module_path)
        return module.main(self.config_class.model_validate(raw_config))


AVAILABLE_MODES: dict[str, type[ProgramMode]] = {}
"""
A registry of all loaded modes.
Key: the mode's name
Value: the mode class
"""


def register_mode(mode: type[ProgramMode]) -> None:
    AVAILABLE_MODES[mode.name] = mode
    logger.info(f"Mode registered: {mode.name}")


ProgramMode.on_init.connect(register_mode)

logger.info("Registering all modes...")
current_dir = os.path.dirname(__file__)
for loader, module_name, is_pkg in pkgutil.iter_modules([current_dir]):
    if not is_pkg:
        continue
    full_module_name = (
        f"{__name__}.{module_name}" if __name__ != "__main__" else module_name
    )
    logger.info(f"Importing module: {full_module_name}")
    importlib.import_module(full_module_name)
