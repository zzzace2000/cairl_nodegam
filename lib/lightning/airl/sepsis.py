"""
Deep Reinforcement Learning: Deep Q-network (DQN)
This example is based on https://github.com/PacktPublishing/Deep-Reinforcement-Learning-Hands-On-
Second-Edition/blob/master/Chapter06/02_dqn_pong.py
The template illustrates using Lightning for Reinforcement Learning. The example builds a basic DQN using the
classic CartPole environment.
To run the template just run:
python reinforce_learn_Qnet.py
After ~1500 steps, you will see the total_reward hitting the max score of 200. Open up TensorBoard to
see the metrics:
tensorboard --logdir default
"""

import argparse

from torch.utils.data import DataLoader
from ...sepsis_simulator.dataset import SepsisExpertDataset
from .base import Base_AIRLLightning, Base_AIRL_NODEGAM_Lightning
from .disc import FCNN_Disc


class SepsisMixin(object):
    monitor_metric = 'val_a'
    monitor_mode = 'max'

    def _dataloader(self, split='train') -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences"""

        dataset = SepsisExpertDataset(
            mdp=self.hparams.mdp,
            N=self.hparams.N,
            gamma=self.hparams.gamma,
            split=split,
            val_ratio=0.2,
            expert_pol=self.hparams.expert_pol,
        )
        dataloader = DataLoader(dataset=dataset,
                                batch_size=self.hparams.batch_size,
                                )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        return self._dataloader('train')

    def val_dataloader(self) -> DataLoader:
        return self._dataloader('val')

    def test_dataloader(self) -> DataLoader:
        return self._dataloader('test')

    @classmethod
    def get_rs_loader(cls, args, rs=None):
        rs = super().get_rs_loader(args, rs=rs)

        # rs.add_rs_hparams('seed', short_name='s', chose_from=[321])
        rs.add_rs_hparams('seed', short_name='s', gen=lambda hparams: rs.np_gen.randint(200))
        rs.add_rs_hparams('batch_size', short_name='bs', chose_from=[512])
        rs.add_rs_hparams('noise', short_name='dn', chose_from=[0., 0.1])
        rs.add_rs_hparams('noise_epochs', short_name='dns',
                          gen=lambda hparams: rs.np_gen.choice([0.1, 0.2]) if hparams.noise > 0 else 0)
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
        parser.add_argument('--batch_size', type=int, default=512)
        parser.add_argument('--gamma', type=float, default=0.9,
                            help='Decay rate in RL. Set it to 0.9 to encourage treating patients '
                                 'earlier to leave the hospitals.')
        parser.add_argument('--noise', type=float, default=0.1)
        parser.add_argument('--noise_epochs', type=float, default=0.4)
        # Environment
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--patience', type=int, default=30)
        parser.add_argument('--mdp', type=str, choices=['original', 'gam', 'linear', 'cgam', 'clinear',
                                                        'cogam', 'colinear'],
                            default='gam', help='How to generate reward.')
        parser.add_argument('--fold', type=int, default=0)
        parser.add_argument('--model_gamma', type=float, default=None,
                            help='The gamma of the model. If None, same as gamma')
        parser.add_argument('--N', type=int, default=5000,
                            help='Number of samples generated')
        parser.add_argument('--expert_pol', type=str, default='optimal',
                            choices=['optimal', 'eps0.07', 'eps0.14'])
        parser = super().add_model_specific_args(parser)
        return parser

    def trainer_args(self):
        return dict()


class AIRLLightning(SepsisMixin, Base_AIRLLightning):
    pass


class AIRL_NODEGAM_Lightning(SepsisMixin, Base_AIRL_NODEGAM_Lightning):
    pass


class AIRL_FCNN_Lightning(SepsisMixin, Base_AIRLLightning):
    disc_model_cls = FCNN_Disc
