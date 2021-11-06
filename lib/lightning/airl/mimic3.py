"""
MIMIC3 Lightning AIRL models. See the base to observe common interfaces
"""

import argparse

from lib.mimic3.dataset import HypotensionDataset, HypotensionWithBCProbDataset
from lib.lightning.airl.disc import MIMIC3_LinearDisc, MIMIC3_NODEGAM_Disc, MIMIC3_FCNN_Disc
from lib.lightning.airl.generator import MIMIC3Generator, MIMIC3BCGenerator
from .base import Base_AIRLLightning, Base_AIRL_NODEGAM_Lightning


class MIMIC3Mixin(object):
    monitor_metric = 'val_a'
    monitor_mode = 'max'
    preprocess = 'quantile'

    def train_dataloader(self):
        # Not very clean here. But we need to change the dataset if the
        # generator wants to regularize kl. Or we can just use the HypoWithBC dataset.
        # But it might be a waste of time if no one is using it
        the_cls = HypotensionDataset
        if self.hparams.get('bc_kl', 0.) > 0.:
            the_cls = HypotensionWithBCProbDataset

        return the_cls.make_loader(
            data_kwargs=dict(
                fold=self.hparams.fold,
                preprocess=self.__class__.preprocess,
            ),
            split='train',
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.workers,
            debug=self.hparams.name.startswith('debug'),
        )

    def val_dataloader(self):
        the_cls = HypotensionDataset
        if self.hparams.get('bc_kl', 0.) > 0.:
            the_cls = HypotensionWithBCProbDataset

        return the_cls.make_loader(
            data_kwargs=dict(
                fold=self.hparams.fold,
                preprocess=self.__class__.preprocess,
            ),
            split='val',
            batch_size=2 * self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.workers,
            debug=self.hparams.name.startswith('debug'),
        )

    def test_dataloader(self):
        the_cls = HypotensionDataset
        if self.hparams.get('bc_kl', 0.) > 0.:
            the_cls = HypotensionWithBCProbDataset

        return the_cls.make_loader(
            data_kwargs=dict(
                fold=self.hparams.fold,
                preprocess=self.__class__.preprocess,
            ),
            split='test',
            batch_size=2 * self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.workers,
            debug=self.hparams.name.startswith('debug'),
        )

    @classmethod
    def get_rs_loader(cls, args, rs=None):
        rs = super().get_rs_loader(args, rs=rs)

        # rs.add_rs_hparams('seed', short_name='s', chose_from=[321])
        rs.add_rs_hparams('seed', short_name='s', gen=lambda hparams: rs.np_gen.randint(200))
        rs.add_rs_hparams('noise', short_name='dn', chose_from=[0, 0.1])
        rs.add_rs_hparams('noise_epochs', short_name='dns', chose_from=[0.8])
        # rs.add_rs_hparams('batch_size', short_name='bs', chose_from=[32])
        return rs

    @classmethod
    def add_model_specific_args(cls, parser) -> argparse.ArgumentParser:
        """
        Adds arguments for DQN model
        Note: these params are fine tuned for Pong env
        Args:
            parent
        """
        # Model
        parser.add_argument('--batch_size', type=int, default=32)
        parser.add_argument('--gamma', type=float, default=1.,
                            help='Decay rate in RL.')
        parser.add_argument('--noise', type=float, default=0.1)
        parser.add_argument('--noise_epochs', type=float, default=0.4)

        # Environment
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--patience', type=int, default=-1)
        parser.add_argument('--workers', type=int, default=0) # If bigger than 0, will crash
        parser.add_argument('--fold', type=int, default=0)
        parser = super().add_model_specific_args(parser)
        return parser

    def trainer_args(self):
        return dict()


class AIRL_MIMIC3_Lightning(MIMIC3Mixin, Base_AIRLLightning):
    disc_model_cls = MIMIC3_LinearDisc
    gen_model_cls = MIMIC3Generator
    preprocess = 'standard'


class AIRL_MIMIC3_NODEGAM_Lightning(MIMIC3Mixin, Base_AIRL_NODEGAM_Lightning):
    disc_model_cls = MIMIC3_NODEGAM_Disc
    gen_model_cls = MIMIC3Generator


class AIRL_MIMIC3_FCNN_Lightning(MIMIC3Mixin, Base_AIRLLightning):
    disc_model_cls = MIMIC3_FCNN_Disc
    gen_model_cls = MIMIC3Generator


## Use exp loss to do early stopping!
class AIRL_MIMIC3_Lightning2(AIRL_MIMIC3_Lightning):
    monitor_metric = 'val_exp_loss'
    monitor_mode = 'min'


class AIRL_MIMIC3_NODEGAM_Lightning2(AIRL_MIMIC3_NODEGAM_Lightning):
    monitor_metric = 'val_exp_loss'
    monitor_mode = 'min'


## The generator is just a bc policy! Also use exp_loss to early stop
class AIRL_MIMIC3_NODEGAM_Lightning3(AIRL_MIMIC3_NODEGAM_Lightning2):
    gen_model_cls = MIMIC3BCGenerator
