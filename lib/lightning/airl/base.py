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
from collections import OrderedDict
from typing import Dict

import pytorch_lightning as pl
import torch

from lib.lightning.airl.disc import LinearDisc, NODEGAM_Disc
from lib.lightning.airl.generator import SepsisGenerator
from lib.random_search import RandomSearch


# from .common import wrappers


class Base_AIRLLightning(pl.LightningModule):
    """ Implement Adversarial Imitation RL for simulated sepsis
    """
    disc_model_cls = LinearDisc
    gen_model_cls = SepsisGenerator

    monitor_metric = 'val_a'
    monitor_mode = 'max'

    def __init__(self, hparams: argparse.Namespace) -> None:
        super().__init__()

        self.save_hyperparameters(hparams)
        if self.hparams.get('model_gamma', None) is None:
            self.hparams.model_gamma = self.hparams.gamma

        ## Discriminator
        self.disc = self.disc_model_cls(self.hparams)

        ## Generator
        self.gen = self.gen_model_cls(self.hparams)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.disc(x)

    def training_step(self, batch, batch_idx, optimizer_idx=0) -> OrderedDict:
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved
        Args:
            batch: current mini batch of expert data
            batch_idx: batch number, not used
        Returns:
            Training loss and log metrics
        """
        # Update generators
        if optimizer_idx == 0:
            name = 'disc'
            loss_dict = self.disc.cal_loss(batch, self.gen, self.current_epoch, self.global_step)
        else:
            name = 'gen'
            loss_dict = self.gen.update(batch=batch, reward_disc=self.disc,
                                        step=self.global_step, epoch=self.current_epoch)

        if loss_dict is None: # Not updating
            return

        if self.trainer.use_dp or self.trainer.use_ddp2:
            loss_dict['loss'] = loss_dict['loss'].unsqueeze(0)

        log = {f'train_{name}_{k}': loss_dict[k] for k in loss_dict}

        status = dict(steps=torch.tensor(self.global_step).to(self.device),
                  **{f'train_{name}_{k}': loss_dict[k] for k in loss_dict})
        self.log_dict(log)

        return OrderedDict({'loss': loss_dict['loss'], 'log': log, 'progress_bar': status})

    def validation_step(self, batch, batch_idx):
        """ Test the action matched and the reward (evaluators) """
        return self._eval_step(batch, batch_idx)

    def validation_epoch_end(self, outputs) -> Dict[str, torch.Tensor]:
        """Log the avg of the test results"""
        return self._eval_epoch_end(outputs, prefix='val')

    def test_step(self, batch, batch_idx):
        """ Test the action matched and the reward (evaluators) """
        return self._eval_step(batch, batch_idx)

    def test_epoch_end(self, outputs) -> Dict[str, torch.Tensor]:
        """Log the avg of the test results"""
        return self._eval_epoch_end(outputs, prefix='test')

    def _eval_step(self, batch, batch_idx):
        """ Test the action matched and the reward (evaluators) """
        result = self.gen.eval_step(batch, batch_idx)
        if hasattr(self.disc, 'eval_step'):
            result.update(self.disc.eval_step(batch, batch_idx))
        return result

    def _eval_epoch_end(self, outputs, prefix='val'):
        """Log the avg of the test results"""
        logs = self.gen.eval_epoch_end(outputs, prefix=prefix)
        if hasattr(self.disc, 'eval_epoch_end'):
            logs.update(self.disc.eval_epoch_end(outputs, prefix=prefix))
        # For callbacks
        self.log_dict(logs)

        result = {'log': logs, 'progress_bar': dict(
            steps=torch.tensor(self.global_step).to(self.device),
            **logs
        )}
        result.update(logs)
        return result

    def configure_optimizers(self):
        """ Initialize Adam optimizer"""
        return [self.disc.optimizer, self.gen.optimizer]

    @classmethod
    def get_rs_loader(cls, args, rs=None):
        if rs is None:
            rs = RandomSearch(hparams=args, seed=args.seed)
        rs = cls.disc_model_cls.get_rs_loader(args, rs=rs)
        rs = cls.gen_model_cls.get_rs_loader(args, rs=rs)
        return rs

    @classmethod
    def add_model_specific_args(cls, parser) -> argparse.ArgumentParser:
        """
        Adds arguments for DQN model
        Note: these params are fine tuned for Pong env
        Args:
            parent
        """
        parser = cls.disc_model_cls.add_model_specific_args(parser)
        parser = cls.gen_model_cls.add_model_specific_args(parser)
        return parser


class Base_AIRL_NODEGAM_Lightning(Base_AIRLLightning):
    disc_model_cls = NODEGAM_Disc

    def training_step(self, batch, batch_idx, optimizer_idx=0) -> OrderedDict:
        self.disc.choice_fn.temp_step_callback(self.global_step)
        return super().training_step(batch, batch_idx, optimizer_idx)

    def on_load_checkpoint(self, checkpoint) -> None:
        self.disc.choice_fn.temp_step_callback(checkpoint['global_step'])
