
from typing import Tuple, List, Dict
import argparse
from collections import OrderedDict
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pytorch_lightning as pl
# from .common import wrappers
from lib.sepsis_simulator.policy import SepsisOptimalSolver
from lib.sepsis_simulator.sepsisSimDiabetes.State import State
from lib.sepsis_simulator.sepsisSimDiabetes.Action import Action
from lib.mimic3.dataset import HypotensionDataset
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from lib.sepsis_simulator.utils import run_policy_to_get_exp
import pickle
import numpy as np
from os.path import exists as pexists
from lib.sepsis_simulator.utils import run_policy
from lib.utils import Timer
from lib.lightning.utils import Squeeze, MinMaxNormalize
from lib.random_search import RandomSearch
from lib import nodegam
from sklearn.metrics import roc_auc_score
from ..utils import evaluating
from lib.lightning.utils import create_fc_layer


class LinearDisc(nn.Module):
    def __init__(self, hparams) -> None:
        '''
        experiment: choose which experiment to run. If select sepsis,
            the the architecture includes a Min-Max normalization layer.
            If select hypotension, then no normalization is performed
        '''
        super().__init__()
        self.hparams = hparams
        self.build()

        # Backward compatability
        # lr = self.hparams.get('disc_lr', self.hparams.get('lr', None))
        # wd = self.hparams.get('disc_wd', self.hparams.get('wd', 0))

        self.optimizer = optim.Adam(self.parameters(), lr=self.hparams.disc_lr,
                                    betas=(0.5, 0.99), weight_decay=self.hparams.disc_wd)

    def build(self) -> None:
        # g is the reward net, and h is the shaping net. Combined would be advantage
        self.g = torch.nn.Sequential(
            MinMaxNormalize(min=State.PHI_MIN, max=State.PHI_MAX),
            torch.nn.Linear(State.PHI_DIM, 1),
            Squeeze(dim=-1),
        )
        if self.hparams.shaping:
            self.h = torch.nn.Sequential(
                MinMaxNormalize(min=State.PHI_MIN, max=State.PHI_MAX),
                torch.nn.Linear(State.PHI_DIM, 1),
                Squeeze(dim=-1),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.g(x)

    def cal_loss(self, expert_batch, generator, epoch, step):
        gen_batch = generator.gen_exp(expert_batch)

        # Concatenate
        cur_device = next(self.parameters()).device
        o_next, o = (
            torch.cat([torch.tensor(gen_batch[k], device=cur_device), expert_batch[k]], dim=0)
            for k in ['o_next', 'o'])

        # Add noise to inputs. Linearly decreasing noise
        def add_noise(x):
            if self.hparams.noise == 0 or self.hparams.noise_epochs == 0:
                return x
            n_ratio = max(1 - (epoch / (self.hparams.noise_epochs * self.hparams.epochs)), 0)
            return x + n_ratio * self.hparams.noise * torch.randn_like(x)

        o_next, o = (add_noise(tmp) for tmp in (o_next, o))

        # Do a special AIRL optimization for D
        # Note D = exp(f) / exp(f) + pi
        # Note D's logits is g(s') + (\gamma * h(s') - h(s)) - logpi(s, a, s')
        reward = self.get_reward(o, o_next)

        np_s, np_a = (np.concatenate([gen_batch[k], expert_batch[k].cpu().numpy()])
                      for k in ['s', 'a'])
        prob = generator.get_action_probs(np_s, np_a)

        logpi = torch.from_numpy(np.log(prob + 1e-8)).to(reward.device)
        # D_logit = reward - prob
        ## Should I use AIRL reward design?
        # D_logit = reward
        return self.cal_loss_by_logits(reward, logpi, epoch)

    def cal_loss_by_logits(self, reward, logpi, epoch):
        D_logit = reward
        if self.hparams.airl_obj > 0.:
            coeff = min(epoch / (self.hparams.airl_anneal_epochs * self.hparams.epochs + 1e-8), 1.)
            D_logit -= coeff * logpi

        num_exp = D_logit.shape[0] // 2
        ## (1) expert loss = -log(D(expert))
        eps = self.hparams.label_smoothing
        exp_logp = (1. - eps) * F.logsigmoid(D_logit[num_exp:]) + eps * F.logsigmoid(-D_logit[num_exp:])
        exp_loss = -torch.sum(exp_logp)
        ## (2) generator loss = log(1 - D(gen)). Don't do smoothing
        gen_logp = F.logsigmoid(-D_logit[:num_exp])
        gen_loss = -torch.sum(gen_logp)

        ## Should I do trick with label smoothing?
        # gen_loss = torch.sum(F.logsigmoid(D_logit[:num_exp]))
        # gen_loss = torch.sum(-F.logsigmoid(-D_logit[:num_exp]))
        loss = (exp_loss + gen_loss) / D_logit.shape[0]

        ## Calculate the AUC
        y_true = np.concatenate([np.zeros(num_exp), np.ones(num_exp)]).astype(int)
        y_score = D_logit.detach().cpu().numpy()
        auc = roc_auc_score(y_true, y_score)

        # exp_acc = (D_logit[num_exp:] >= 0).sum() / num_exp
        # gen_acc = (D_logit[:num_exp] < 0).sum() / num_exp]

        return {'loss': loss, 'auc': auc}

    def get_reward(self, s, s_next, dones=0):
        ''' Return the reward of shape B instead of (B, 1) '''
        the_s = s_next if self.hparams.disc_state_time == 'next' else s

        reward = self.g(the_s)
        if self.hparams.shaping:
            shaping = ((1 - dones) * self.hparams.model_gamma * self.h(s_next) - self.h(s))
            reward += shaping

        return reward

    @classmethod
    def get_rs_loader(cls, args, rs=None):
        if rs is None:
            rs = RandomSearch(hparams=args, seed=args.seed)
        rs.add_rs_hparams('disc_lr', short_name='dlr', chose_from=[2e-4, 4e-4])
        # rs.add_rs_hparams('airl_obj', short_name='aob', chose_from=[0, 1])
        rs.add_rs_hparams('shaping', short_name='sh', chose_from=[0, 1])
        # rs.add_rs_hparams('label_smoothing', short_name='ls', chose_from=[0, 0.05])
        return rs

    @classmethod
    def add_model_specific_args(cls, parser) -> argparse.ArgumentParser:
        """
        Adds arguments for DQN model
        Note: these params are fine tuned for Pong env
        Args:
            parent
        """
        # Optimization
        parser.add_argument('--disc_lr', type=float, default=3e-4)
        parser.add_argument('--disc_wd', default=0., type=float)

        # Objective
        parser.add_argument('--shaping', type=int, default=1,
                            help='Model shaping reward or not.')
        parser.add_argument("--label_smoothing", type=float, default=0.)
        parser.add_argument('--airl_obj', type=float, default=0,
                            help='Use AIRL objective or not')
        parser.add_argument('--airl_anneal_epochs', type=float, default=0.5,
                            help='Annealing the airl objective. 0 means no annealing'
                                 'and 10 means increase to 1 in first 10 epochs')
        parser.add_argument("--disc_state_time", type=str, default='next',
                            choices=['current', 'next'])
        return parser

    def temp_callback(self, global_step):
        pass


class NODEGAM_Disc(LinearDisc):
    def build(self) -> None:
        self.choice_fn = nodegam.EM15Temp(
            max_temp=1., min_temp=0.01, steps=self.hparams.anneal_steps)

        the_dict = dict(
            input_dim=State.PHI_DIM,
            layer_dim=self.hparams.num_trees,
            num_layers=self.hparams.num_layers,
            num_classes=1,
            addi_tree_dim=self.hparams.addi_tree_dim,
            depth=self.hparams.depth,
            flatten_output=False,
            choice_function=self.choice_fn,
            bin_function=nodegam.entmoid15,
            output_dropout=self.hparams.output_dropout,
            last_dropout=self.hparams.last_dropout,
            colsample_bytree=self.hparams.colsample_bytree,
            add_last_linear=self.hparams.add_last_linear,
        )

        self.g = torch.nn.Sequential(
            # Normalize the input features
            MinMaxNormalize(min=State.PHI_MIN, max=State.PHI_MAX),
            nodegam.GAMBlock(**the_dict),
        )
        if self.hparams.shaping:
            self.h = torch.nn.Sequential(
                MinMaxNormalize(min=State.PHI_MIN, max=State.PHI_MAX),
                nodegam.GAMBlock(**the_dict),
            )

    @classmethod
    def get_rs_loader(cls, args, rs=None):
        rs = super().get_rs_loader(args, rs=rs)
        rs.add_rs_hparams('num_layers', short_name='nl', chose_from=[1, 2])
        rs.add_rs_hparams('num_trees', short_name='nt', chose_from=[100, 200, 400])
        rs.add_rs_hparams('addi_tree_dim', short_name='ad', chose_from=[0, 1])
        rs.add_rs_hparams('depth', short_name='td', chose_from=[1, 2])
        rs.add_rs_hparams('output_dropout', short_name='od', chose_from=[0., 0.1])
        rs.add_rs_hparams('last_dropout', short_name='ld', chose_from=[0., 0.3])
        rs.add_rs_hparams('colsample_bytree', short_name='cs', chose_from=[1., 0.5])
        rs.add_rs_hparams('anneal_steps', short_name='an', chose_from=[3000])
        # rs.add_rs_hparams('l2_lambda', short_name='la', chose_from=[10, 20])
        return rs

    @classmethod
    def add_model_specific_args(cls, p):
        p = super().add_model_specific_args(p)
        # p.add_argument("--arch", type=str, default='GAM',
        #                choices=['ODST', 'GAM', 'GAMAtt', 'GAMAtt3'])
        p.add_argument("--anneal_steps", type=int, default=500)
        p.add_argument("--num_trees", type=int, default=1024)
        p.add_argument("--num_layers", type=int, default=2)
        p.add_argument("--addi_tree_dim", type=int, default=0)
        p.add_argument("--depth", type=int, default=1)
        p.add_argument("--output_dropout", type=float, default=0.)
        p.add_argument("--l2_lambda", type=float, default=0.)
        p.add_argument("--last_dropout", type=float, default=0.)
        p.add_argument("--colsample_bytree", type=float, default=1.)
        p.add_argument("--add_last_linear", type=int, default=1)
        return p


class FCNN_Disc(LinearDisc):
    def build(self) -> None:
        self.g = torch.nn.Sequential(
            # Normalize the input features
            MinMaxNormalize(min=State.PHI_MIN, max=State.PHI_MAX),
            create_fc_layer(
                n_in=State.PHI_DIM,
                n_layer=self.hparams.disc_n_layer,
                n_hidden=self.hparams.disc_n_hidden,
                n_out=1,
                dropout=self.hparams.disc_dropout,
            ),
            Squeeze(dim=-1),
        )
        if self.hparams.shaping:
            self.h = torch.nn.Sequential(
                # Normalize the input features
                MinMaxNormalize(min=State.PHI_MIN, max=State.PHI_MAX),
                create_fc_layer(
                    n_in=State.PHI_DIM,
                    n_layer=self.hparams.disc_n_layer,
                    n_hidden=self.hparams.disc_n_hidden,
                    n_out=1,
                    dropout=self.hparams.disc_dropout,
                ),
                Squeeze(dim=-1),
            )

    @classmethod
    def get_rs_loader(cls, args, rs=None):
        rs = super().get_rs_loader(args, rs=rs)
        rs.add_rs_hparams('disc_n_layer', short_name='dnl', chose_from=[2, 3, 4, 5])
        rs.add_rs_hparams('disc_n_hidden', short_name='dnh', chose_from=[32, 64, 128, 256])
        rs.add_rs_hparams('disc_dropout', short_name='ddr', chose_from=[0.1, 0.3, 0.5])
        return rs

    @classmethod
    def add_model_specific_args(cls, p):
        p = super().add_model_specific_args(p)
        p.add_argument("--disc_n_layer", type=int, default=2)
        p.add_argument("--disc_n_hidden", type=int, default=64)
        p.add_argument("--disc_dropout", type=float, default=0.5)
        return p


'''
Below are the discriminators for MIMIC3 datasets
'''
class MIMIC3_LinearDisc(LinearDisc):
    def __init__(self, hparams):
        # Add a discriminator type input
        # if 'disc_state_type' not in hparams:
        #     hparams['disc_state_type'] = 'states'

        self.input_dim = HypotensionDataset.get_state_dim_by_type(hparams['disc_state_type'])
        super().__init__(hparams)

    def build(self) -> None:
        # g is the reward net, and h is the shaping net. Combined would be advantage
        self.g = torch.nn.Sequential(
            torch.nn.Linear(self.input_dim, 1),
            Squeeze(dim=-1),
        )
        if self.hparams.shaping:
            self.h = torch.nn.Sequential(
                torch.nn.Linear(self.input_dim, 1),
                Squeeze(dim=-1),
            )

    def cal_loss(self, expert_batch, generator, epoch, step):
        with torch.no_grad(), evaluating(generator):
            gen_s, gen_a, gen_next_s, gen_dones = generator.gen_exp(
                expert_batch, actions='agent', state_type='all',
                same_act_use_exp=self.hparams.same_act_use_exp)

            exp_s, exp_a, exp_next_s, exp_dones = generator.gen_exp(
                expert_batch, actions='expert', state_type='all',
                same_act_use_exp=self.hparams.same_act_use_exp)

        s = torch.cat([gen_s, exp_s], dim=0)
        a = torch.cat([gen_a, exp_a], dim=0)
        next_s = torch.cat([gen_next_s, exp_next_s], dim=0)
        dones = torch.cat([gen_dones, exp_dones], dim=0)

        # Generator takes all features to generate log probs...
        with torch.no_grad(), evaluating(generator):
            logpi = generator.get_action_log_probs(s, a)

        # Add noise to inputs. Linearly decreasing noise
        def add_noise(x):
            if self.hparams.noise == 0 or self.hparams.noise_epochs == 0:
                return x
            n_ratio = max(1 - (epoch / (self.hparams.noise_epochs * self.hparams.epochs + 1e-8)), 0)
            return x + n_ratio * self.hparams.noise * torch.randn_like(x)

        s, next_s = (add_noise(tmp) for tmp in (s, next_s))
        reward = self.get_reward(s, next_s, dones)

        return self.cal_loss_by_logits(reward, logpi, epoch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] != self.input_dim:
            x = HypotensionDataset.extract_s_by_state_type(x, self.hparams.disc_state_type)
        reward = self.g(x)
        return reward

    def get_reward(self, s, s_next, dones=0, shaping=None):
        ''' Return the reward of shape B instead of (B, 1) '''
        s = HypotensionDataset.extract_s_by_state_type(s, self.hparams.disc_state_type)
        s_next = HypotensionDataset.extract_s_by_state_type(s_next, self.hparams.disc_state_type)

        the_s = s_next if self.hparams.disc_state_time == 'next' else s
        reward = self.g(the_s)

        if shaping is None:
            shaping = self.hparams.shaping

        if shaping:
            next_V = self.hparams.model_gamma * self.h(s_next)
            next_V[dones] = 0.
            shaping = (next_V - self.h(s))
            reward += shaping

        return reward

    def eval_step(self, batch, batch_idx):
        states, actions, next_states, dones = HypotensionDataset.extract_s_and_a_pairs(batch['x_list'])

        D_logit = self.get_reward(states, next_states, dones)

        exp_logp = F.logsigmoid(D_logit)
        exp_loss = -torch.sum(exp_logp)
        exp_acc = torch.sum((D_logit > 0).float())
        return {f'exp_loss': exp_loss, 'exp_acc': exp_acc, f'total': states.shape[0]}

    def eval_epoch_end(self, outputs, prefix='val'):
        """Log the avg of the test results"""
        exp_loss = sum([x['exp_loss'] for x in outputs])
        exp_acc = sum([x['exp_acc'] for x in outputs])
        total = sum([x['total'] for x in outputs])

        avg_exp_loss = exp_loss / total
        avg_exp_acc = exp_acc / total

        logs = {f'{prefix}_exp_loss': avg_exp_loss, f'{prefix}_exp_acc': avg_exp_acc}
        return logs

    @classmethod
    def get_rs_loader(cls, args, rs=None):
        if rs is None:
            rs = RandomSearch(hparams=args, seed=args.seed)
        rs.add_rs_hparams('disc_lr', short_name='dlr', chose_from=[5e-4, 8e-4, 1e-3])
        # rs.add_rs_hparams('disc_wd', short_name='dwd', chose_from=[0.])
        # rs.add_rs_hparams('shaping', short_name='shp', chose_from=[0, 1])
        rs.add_rs_hparams('airl_obj', short_name='aob', chose_from=[0, 1])

        rs.add_rs_hparams('label_smoothing', short_name='ls', chose_from=[0.005, 0.01])
        # rs.add_rs_hparams('same_act_use_exp', short_name='sa', chose_from=[0, 0.5, 1])
        # rs.add_rs_hparams('disc_state_type', short_name='', chose_from=['features'])
        return rs

    @classmethod
    def add_model_specific_args(cls, parser) -> argparse.ArgumentParser:
        parser = super().add_model_specific_args(parser)
        parser.add_argument("--disc_state_type", type=str, default='features',
                            choices=['features', 'states', 'all'])
        parser.add_argument("--same_act_use_exp", type=float, default=1.,
                            help='If 1, use orig exp to distinguish. If 0, use gru pred. '
                                 'It may increase difficulty.')
        return parser

    def temp_callback(self, global_step):
        pass


class MIMIC3_NODEGAM_Disc(MIMIC3_LinearDisc):
    def build(self) -> None:
        self.choice_fn = nodegam.EM15Temp(
            max_temp=1., min_temp=0.01, steps=self.hparams.anneal_steps)

        the_dict = dict(
            input_dim=self.input_dim,
            layer_dim=self.hparams.num_trees,
            num_layers=self.hparams.num_layers,
            num_classes=1,
            addi_tree_dim=self.hparams.addi_tree_dim,
            depth=self.hparams.depth,
            flatten_output=False,
            choice_function=self.choice_fn,
            bin_function=nodegam.entmoid15,
            output_dropout=self.hparams.output_dropout,
            last_dropout=self.hparams.last_dropout,
            colsample_bytree=self.hparams.colsample_bytree,
            add_last_linear=self.hparams.add_last_linear,
            ga2m=self.hparams.ga2m,
        )
        if self.hparams.disc_arch.startswith('GAMAtt'):
            the_dict['dim_att'] = self.hparams.dim_att

        arch_cls = getattr(nodegam, self.hparams.disc_arch + 'Block')
        self.g = arch_cls(**the_dict)
        if self.hparams.shaping:
            self.h = arch_cls(**the_dict)

    @classmethod
    def get_rs_loader(cls, args, rs=None):
        rs = super().get_rs_loader(args, rs=rs)
        # rs.add_rs_hparams('airl_anneal_epochs', short_name='ane',
        #                   gen=lambda hparams: -1 if hparams.airl_obj == 0
        #                   else rs.np_gen.choice([0.5]))
        # rs.add_rs_hparams('ga2m', short_name='ga2m', chose_from=[0, 1])
        rs.add_rs_hparams('disc_arch', short_name='a', chose_from=['GAM'])
        rs.add_rs_hparams('num_layers', short_name='nl', chose_from=[1, 2, 3])
        rs.add_rs_hparams('num_trees', short_name='nt',
                          chose_from=[200, 300])
                          # gen=(lambda args: rs.np_gen.choice(
                          #      [500, 1000] if args.num_layers >= 4 else [500, 1000])))
        rs.add_rs_hparams('addi_tree_dim', short_name='ad', #chose_from=[0, 1],
                          gen=lambda args: rs.np_gen.choice([0, 1]) if args.num_layers > 1 else 0
                          )
        rs.add_rs_hparams('depth', short_name='td', #chose_from=[2, 3],
                          gen=lambda args: rs.np_gen.choice([2, 3, 4])
                          )
        rs.add_rs_hparams('last_dropout', short_name='ld', chose_from=[0.3, 0.5])
        rs.add_rs_hparams('colsample_bytree', short_name='cs', chose_from=[0.5])
        rs.add_rs_hparams('anneal_steps', short_name='an', chose_from=[3000])
        rs.add_rs_hparams('l2_lambda', short_name='la', chose_from=[0])
        rs.add_rs_hparams('add_last_linear', short_name='ll',
                          gen=lambda args: 0 if args.disc_arch == 'GAMAtt' else rs.np_gen.choice([1]))
        rs.add_rs_hparams('output_dropout', short_name='od',
                          gen=lambda args: 0 if args.add_last_linear == 0 else rs.np_gen.choice([0.1, 0.2]))
        # rs.add_rs_hparams('dim_att', short_name='da',
        #                   gen=lambda args: rs.np_gen.choice([8, 16, 32])
        #                   if args.disc_arch.startswith('GAMAtt') else -1)
        return rs

    @classmethod
    def add_model_specific_args(cls, p):
        p = super().add_model_specific_args(p)
        p.add_argument("--disc_arch", type=str, default='GAM',
                       choices=['ODST', 'GAM', 'GAMAtt', 'GAMAtt3'])
        p.add_argument("--anneal_steps", type=int, default=500)
        p.add_argument("--num_trees", type=int, default=128)
        p.add_argument("--num_layers", type=int, default=2)
        p.add_argument("--addi_tree_dim", type=int, default=0)
        p.add_argument("--depth", type=int, default=2)
        p.add_argument("--output_dropout", type=float, default=0.)
        p.add_argument("--l2_lambda", type=float, default=0.)
        p.add_argument("--last_dropout", type=float, default=0.)
        p.add_argument("--colsample_bytree", type=float, default=1.)
        p.add_argument("--add_last_linear", type=int, default=1)
        p.add_argument("--ga2m", type=int, default=1)
        p.add_argument("--dim_att", type=int, default=8)
        return p


class MIMIC3_FCNN_Disc(MIMIC3_LinearDisc):
    def build(self) -> None:
        self.g = torch.nn.Sequential(
            # Normalize the input features
            create_fc_layer(
                n_in=self.input_dim,
                n_layer=self.hparams.disc_n_layer,
                n_hidden=self.hparams.disc_n_hidden,
                n_out=1,
                dropout=self.hparams.disc_dropout,
            ),
            Squeeze(dim=-1),
        )
        if self.hparams.shaping:
            self.h = torch.nn.Sequential(
                # Normalize the input features
                create_fc_layer(
                    n_in=self.input_dim,
                    n_layer=self.hparams.disc_n_layer,
                    n_hidden=self.hparams.disc_n_hidden,
                    n_out=1,
                    dropout=self.hparams.disc_dropout,
                ),
                Squeeze(dim=-1),
            )

    @classmethod
    def get_rs_loader(cls, args, rs=None):
        rs = super().get_rs_loader(args, rs=rs)
        rs.add_rs_hparams('disc_n_layer', short_name='dnl', chose_from=[2, 3, 4, 5])
        rs.add_rs_hparams('disc_n_hidden', short_name='dnh', chose_from=[32, 64, 128, 256])
        rs.add_rs_hparams('disc_dropout', short_name='ddr', chose_from=[0.1, 0.3, 0.5])
        return rs

    @classmethod
    def add_model_specific_args(cls, p):
        p = super().add_model_specific_args(p)
        p.add_argument("--disc_n_layer", type=int, default=2)
        p.add_argument("--disc_n_hidden", type=int, default=64)
        p.add_argument("--disc_dropout", type=float, default=0.5)
        return p
