from pytorch_lightning.callbacks.model_checkpoint import ModelCheckpoint
from os.path import exists as pexists, join as pjoin, islink as pislink, basename as pbasename
import os


class MyModelCheckpoint(ModelCheckpoint):
    def __init__(self, *args, notsave_epochs=0, **kwargs):
        super().__init__(*args, **kwargs)
        self.notsave_epochs = notsave_epochs

    def check_monitor_top_k(self, trainer, current=None) -> bool:
        if trainer.current_epoch < self.notsave_epochs:
            return False
        return super().check_monitor_top_k(trainer, current)

    def on_fit_end(self, trainer, pl_module) -> None:
        ''' Create a sym link of which model is the best '''
        best_link_path = pjoin('logs', pl_module.hparams.name, 'best.ckpt')
        if pislink(best_link_path):
            print('WARNING! The best.ckpt already exists before training ends.')
            os.unlink(pjoin('logs', pl_module.hparams.name, 'best.ckpt'))

        os.symlink(pbasename(self.best_model_path), best_link_path)
