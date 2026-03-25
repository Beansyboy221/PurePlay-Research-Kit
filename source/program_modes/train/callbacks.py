import lightning.pytorch.callbacks
import optuna

from globals.enums import binds
from utilities.app_utils import global_logger
from utilities.data_utils import poll_helpers

class KillTrainingCallback(lightning.pytorch.callbacks.Callback):
    '''Kills current optuna study/trial upon pressing kill bind.'''
    def __init__(
            self, 
            kill_binds: list[binds.DigitalBind], 
            kill_bind_gate: binds.BindGate = binds.BindGate.ANY
        ):
        super().__init__()
        self.kill_binds = kill_binds
        self.kill_bind_gate = kill_bind_gate
        self.stopping_flag = False

    def on_train_batch_end(self, trainer, *args, **kwargs):
        '''Pytorch Lightning hook to stop mid-trial.'''
        if not self.stopping_flag:
            if not poll_helpers.should_kill(self.kill_binds, self.kill_bind_gate):
                return
            global_logger.info('Kill bind(s) detected. Stopping...')
            self.stopping_flag = True
            trainer.should_stop = True

    def __call__(
            self, 
            study: optuna.study.Study
        ) -> None:
        '''Optuna hook: Stops the study after the objective function returns.'''
        if self.stopping_flag:
            study.stop()