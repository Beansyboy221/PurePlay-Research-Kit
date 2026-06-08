import logging

import lightning.pytorch.callbacks
import optuna

from beaninput import helpers, binds, config

logger = logging.getLogger()


class KillTrainingCallback(lightning.pytorch.callbacks.Callback):
    """Kills current optuna study/trial upon pressing kill bind."""

    def __init__(
        self,
        kill_binds: list[binds.Bind],
        kill_bind_gate: config.GateCallable = any,
    ):
        super().__init__()
        self.kill_binds = kill_binds
        self.kill_bind_gate = kill_bind_gate
        self.stopping_flag = False

    def on_train_batch_end(self, trainer: lightning.Trainer):
        """Pytorch Lightning hook to stop mid-trial."""
        if not self.stopping_flag:
            if not helpers.are_active(self.kill_binds, self.kill_bind_gate):
                return
            logger.info("Kill bind(s) detected. Stopping...")
            self.stopping_flag = True
            trainer.should_stop = True

    def __call__(self, study: optuna.study.Study) -> None:
        """Optuna hook: Stops the study after the objective function returns."""
        if self.stopping_flag:
            study.stop()
