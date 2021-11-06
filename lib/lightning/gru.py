#!/usr/bin/env python3
# coding=utf-8
"""
A potential tutorial for GRUCell
https://towardsdatascience.com/encoder-decoder-model-for-multistep-time-series-forecasting-using-pytorch-5d54c6af6e60
"""
import copy

import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from ..mimic3.dataset import HypotensionDataset, HypotensionWithBCProbDataset
from ..random_search import RandomSearch
from .utils import create_fc_layer


class HypotensionGRULightning(pl.LightningModule):
    monitor_metric = 'val_loss'
    monitor_mode = 'min'

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.init_model()

    def init_model(self):
        # Initialize the action embedding
        act_in_dim = HypotensionDataset.NUM_FLUID_BINS + HypotensionDataset.NUM_VASO_BINS

        self.act_embedding = nn.Identity()
        if self.hparams.act_n_layer > 0:
            self.act_embedding = create_fc_layer(
                n_in=act_in_dim,
                n_layer=self.hparams.act_n_layer,
                n_hidden=self.hparams.act_n_hidden,
                n_out=self.hparams.act_n_out,
                dropout=self.hparams.act_dropout,
                is_3d_input=True,
            )
            act_in_dim = self.hparams.act_n_out

        self.gru = nn.GRU(
            len(HypotensionDataset.all_cols) + act_in_dim,
            self.hparams.n_hidden,
            self.hparams.n_layer,
            batch_first=True,
            dropout=self.hparams.dropout,
        )

        # Arch for final MLP: with BN, Dropout and ELU...
        self.out = create_fc_layer(
            n_in=self.hparams.n_hidden,
            n_layer=self.hparams.fc_n_layer,
            n_hidden=self.hparams.fc_n_hidden,
            n_out=2 * HypotensionDataset.len_f if self.hparams.obj == 'gaussian'
                else HypotensionDataset.len_f,
            dropout=self.hparams.fc_dropout,
            in_bn=True,
            is_3d_input=True,
        )

    def forward(self, x):
        raise NotImplemented('Do not call this')

    def unzip(self, output):
        num_f = output.shape[-1] // 2
        return output[..., :num_f], output[..., num_f:]

    def simulate(self, exp_states, actions_fn=None, rollout=False, sample_stdev=0.):
        '''
        exp_states: [B, T, D] the expert states
        actions_fn: take in cur obs and gru hidden state up to t-1
            and output action indexes
        sample_stdev: when sampling from Gaussian, what is the stdev ratio from Gaussian.
            If 0., just return mean. If 1, just sample from normal Gaussian.
            Usually in test time we want it to be < 1 to get more-likely outcomes like in
            VAE we sample from the prior < 1.
        '''
        hiddens = None

        if actions_fn is None:
            exp_act = HypotensionDataset.extract_cur_a(exp_states, form='twohot')

        states = exp_states.clone()
        T = exp_states.shape[1]
        preds = []
        for t in range(T-1):
            # Run through DQN to get actions
            if actions_fn is None:
                cur_act = exp_act[:, t, :]
            else:
                cur_act_idx = actions_fn(states[:, t, :])
                cur_act = HypotensionDataset.convert_act_idx_to_twohot(cur_act_idx)

            # Convert into 3-d to pass through MLP...
            cur_act_emb = self.act_embedding(cur_act.unsqueeze(dim=1)).squeeze_(dim=1)

            cur_s_and_a = torch.cat([states[:, t, :], cur_act_emb], dim=-1).unsqueeze(dim=1)
            out, hiddens = self.gru(cur_s_and_a, hx=hiddens)
            pred = self.out(out)

            val = pred[:, 0, :] # for l1 or l2 loss
            if 'obj' in self.hparams and self.hparams.obj == 'gaussian':
                mu, logvar = self.unzip(val)
                val = self.reparametrize(mu, logvar, sample_stdev=sample_stdev)

            preds.append(val)
            if rollout:
                ### Modify next-state
                states[:, (t+1), HypotensionDataset.f_idxes] = val
                # Copy and modify next-state summary actions
                states[:, (t+1), HypotensionDataset.f_actions_idxes] = \
                    HypotensionDataset.update_act_summary(states[:, :(t+1)], cur_act)[:, HypotensionDataset.f_actions_idxes]
        if rollout:
            return states

        preds = torch.stack(preds, dim=1)
        states[:, 1:, HypotensionDataset.f_idxes] = preds
        return states

    @staticmethod
    def reparametrize(mu, logvar, sample_stdev=1.):
        std = logvar.mul(0.5).exp_()

        eps = sample_stdev * std.new(std.size()).normal_()
        return eps.mul(std).add_(mu)

    def training_step(self, batch, batch_idx, hiddens=None):
        ret_dict = self._step(batch, hiddens, is_training=True)

        avg_reg_loss = (ret_dict['sum_reg_loss'] / ret_dict['num_reg'])
        loss = avg_reg_loss
        # alpha = self.hparams.reg_alpha
        # loss = alpha * avg_reg_loss + (1. - alpha) * avg_ind_loss

        logs = {'train_loss': loss, 'train_reg_loss': avg_reg_loss}
        self.log_dict(logs)

        result = {'loss': loss, 'log': logs, 'progress_bar': logs}
        return result

    def validation_step(self, batch, batch_nb):
        print('Come in')
        batch_cp = copy.deepcopy(batch)
        del batch
        return self._step(batch_cp)

    def test_step(self, batch, batch_nb):
        return self._step(batch)

    def _step(self, batch, hiddens=None, is_training=False):
        x_list = batch['x_list']

        x_len = [v.size(0) for v in x_list]
        x_pad = pad_sequence(x_list, batch_first=True)

        # Get current action in each x
        cur_act = HypotensionDataset.extract_cur_a(x_pad, form='twohot')
        cur_act = self.act_embedding(cur_act)

        # Append a last time dimension
        cur_act = torch.cat([cur_act, cur_act.new_zeros(cur_act.shape[0], 1, cur_act.shape[2])], dim=1)
        x_pad = torch.cat([x_pad, cur_act], dim=-1)

        # Just change to always use one_step_pred
        preds = self.one_step_ahead_pred(x_pad, x_len, hiddens=hiddens)

        propensity = None
        if 'bc_prob' in batch and self.hparams.iptw:
            cur_act_idx = HypotensionDataset.extract_cur_a(x_pad, form='act_idx')
            bc_prob = pad_sequence(batch['bc_prob'], batch_first=True)
            bc_prob = bc_prob[:, :-1, :]
            propensity = bc_prob.gather(-1, cur_act_idx.unsqueeze(-1)).squeeze(-1)

            # Calculate marginal probability to calculate stabilized weights
            marginal_p = HypotensionWithBCProbDataset.get_marginal_prob(self.device)
            tmp = marginal_p.unsqueeze(0).unsqueeze(0).expand(*cur_act_idx.shape[:2], marginal_p.shape[0])
            marginal = tmp.gather(-1, cur_act_idx.unsqueeze(-1)).squeeze(-1)

        return self.cal_loss(preds, x_pad, propensity, marginal)

    def one_step_ahead_pred(self, x_pad, x_len, hiddens=None):
        # Implement BPTT here if needed
        x_packed = pack_padded_sequence(
            x_pad, x_len, enforce_sorted=False, batch_first=True)
        out, hiddens = self.gru(x_packed, hiddens)
        out_padded, _ = pad_packed_sequence(out, batch_first=True)
        # ^-- [batch_size, max_len, hidden dim]
        pred = self.out(out_padded)
        return pred[:, :-1, :]  # Ignore the last pred

    def rollout(self, x_pad, tf_ratio=0.):
        preds = []
        hiddens = None
        in_data = x_pad[:, 0:1, :].clone()  # First time step
        for t in range(x_pad.shape[1] - 1):
            out, hiddens = self.gru(in_data, hx=hiddens)
            pred = self.out(out)
            preds.append(pred)

            val = pred  # for l1 or l2 loss
            if self.hparams.obj == 'gaussian':
                mu, logvar = self.unzip(pred)
                val = self.reparametrize(mu, logvar)

            if tf_ratio > 0:
                cond = torch.bernoulli(
                    self.hparams.tf_ratio * torch.ones(*val.shape, device=self.device))
                val = torch.where(
                    cond == 1, x_pad[:, (t + 1):(t + 2), HypotensionDataset.f_idxes], val)

            if self.hparams.bptt_steps > 0 and t % self.hparams.bptt_steps == 0:
                hiddens = hiddens.detach()
                val = val.detach()

            in_data = x_pad[:, (t + 1):(t + 2), :].clone()  # Later time step
            in_data[:, :, HypotensionDataset.f_idxes] = val

        preds = torch.cat(preds, dim=1)
        return preds

    def cal_loss(self, pred, x_pad, propensity=None, marginal=None):
        '''
        pred has only T-1 time steps, while x_pad has T time steps
        '''
        y = x_pad[:, 1:, HypotensionDataset.f_idxes]
        # Note ind is also padded with 0, so rest of dim isn't included in loss
        ind = x_pad[:, 1:, HypotensionDataset.f_ind_idxes]

        # Loss averaged per-prediction
        ret_dict = {}
        if self.hparams.obj == 'gaussian':
            mu, logvar = self.unzip(pred)
            # Calculate unnormalized prediction Gaussian loss
            loss = (0.5 * (logvar + (y - mu) ** 2 * torch.exp(-logvar))) * ind
        elif self.hparams.obj == 'l2':
            loss = F.mse_loss(pred, y, reduction='none') * ind
        else: # smooth_l1
            loss = F.smooth_l1_loss(pred, y, reduction='none') * ind

        num_reg = torch.sum(ind)
        if propensity is not None:
            # To zero-out the padding time, we check if the sum of ind is zero
            is_pad = (torch.sum(ind, dim=-1) == 0).float()
            weights = (marginal / (propensity + 1e-4)) * (1. - is_pad)

            loss *= weights.unsqueeze(-1)
            num_reg = torch.sum(ind * weights.unsqueeze(-1))

        loss = torch.sum(loss)
        ret_dict['sum_reg_loss'] = loss
        ret_dict['num_reg'] = num_reg
        return ret_dict

    def validation_epoch_end(self, outputs):
        return self._epoch_end(outputs, prefix='val')

    def test_epoch_end(self, outputs):
        return self._epoch_end(outputs, prefix='test')

    def _epoch_end(self, outputs, prefix='val'):
        def get_loss(name='reg'):
            all_loss = torch.stack([x[f'sum_{name}_loss'] for x in outputs])
            all_num = torch.stack([x[f'num_{name}'] for x in outputs])
            avg_loss = torch.sum(all_loss) / torch.sum(all_num)
            return avg_loss

        avg_reg_loss = get_loss('reg')

        tensorboard_logs = {f'{prefix}_loss': avg_reg_loss}
        self.log_dict(tensorboard_logs)

        result = {'log': tensorboard_logs}
        result.update(tensorboard_logs)
        return result

    def configure_optimizers(self):  # REQUIRED
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)

    def train_dataloader(self):
        cls = HypotensionWithBCProbDataset if self.hparams.iptw else HypotensionDataset
        return cls.make_loader(
            split='train',
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=self.hparams.get('workers', 0),
            debug=self.hparams.name.startswith('debug'),
        )

    def val_dataloader(self):
        cls = HypotensionWithBCProbDataset if self.hparams.iptw else HypotensionDataset
        return cls.make_loader(
            split='val',
            batch_size=2 * self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.get('workers', 0),
            debug=self.hparams.name.startswith('debug'),
        )

    def test_dataloader(self):
        cls = HypotensionWithBCProbDataset if self.hparams.iptw else HypotensionDataset
        return cls.make_loader(
            split='test',
            batch_size=2 * self.hparams.batch_size,
            shuffle=False,
            num_workers=self.hparams.get('workers', 0),
            debug=self.hparams.name.startswith('debug'),
        )

    @classmethod
    def get_rs_loader(cls, args):
        rs = RandomSearch(hparams=args, seed=args.seed)
        # rs.add_rs_hparams('seed', short_name='s', chose_from=[321])
        rs.add_rs_hparams('seed', short_name='s', gen=lambda hparams: rs.np_gen.randint(200))
        rs.add_rs_hparams('lr', chose_from=[5e-4, 1e-3])
        rs.add_rs_hparams('wd', short_name='wd', chose_from=[0., 1e-5])
        rs.add_rs_hparams('batch_size', short_name='bs', chose_from=[128, 256])
        rs.add_rs_hparams('n_hidden', short_name='nh', chose_from=[64])
        rs.add_rs_hparams('n_layer', short_name='nl', chose_from=[1])
        rs.add_rs_hparams('dropout', short_name='dr',
                          gen=lambda hparams: 0. if hparams.n_layer <= 1 else rs.np_gen.choice([0.3, 0.5]))
        rs.add_rs_hparams('fc_n_hidden', short_name='fnh', chose_from=[256, 384, 512])
        rs.add_rs_hparams('fc_n_layer', short_name='fnl', chose_from=[2])
        rs.add_rs_hparams('fc_dropout', short_name='fdr', chose_from=[0.15]) # better than 0.3
        rs.add_rs_hparams('act_n_hidden', short_name='anh',
                          gen=lambda hparams: 0 if hparams.act_n_layer <= 1 else rs.np_gen.choice([64, 128]))
        rs.add_rs_hparams('act_n_layer', short_name='anl', chose_from=[0, 1, 2]) # No 4
        rs.add_rs_hparams('act_n_out', short_name='ano', chose_from=[32, 64, 96])
        rs.add_rs_hparams('act_dropout', short_name='adr',
                          gen=lambda hparams: 0. if hparams.act_n_layer <= 1 else rs.np_gen.choice([0.3]))
        # rs.add_rs_hparams('tf_epochs', short_name='tfe', chose_from=[10, 20, 40])
        # rs.add_rs_hparams('tf_ratio', short_name='tfr', chose_from=[0.])
        return rs

    @staticmethod
    def add_model_specific_args(parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser.add_argument('--workers', type=int, default=0)
        parser.add_argument('--epochs', type=int, default=300)
        parser.add_argument('--patience', type=int, default=50)
        parser.add_argument('--lr', default=1e-3, type=float)
        parser.add_argument('--wd', default=0., type=float)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--n_hidden', default=16, type=int)
        parser.add_argument('--n_layer', default=1, type=int)
        parser.add_argument('--dropout', default=0., type=float)
        parser.add_argument('--fc_n_hidden', default=16, type=int)
        parser.add_argument('--fc_n_layer', default=2, type=int)
        parser.add_argument('--fc_dropout', default=0.1, type=float)
        parser.add_argument('--act_n_hidden', default=16, type=int)
        parser.add_argument('--act_n_layer', default=2, type=int)
        parser.add_argument('--act_n_out', default=16, type=int)
        parser.add_argument('--act_dropout', default=0.5, type=float)
        parser.add_argument('--bptt_steps', default=0, type=int)
        # Teacher forcing flag......
        # parser.add_argument('--tf_epochs', default=1, type=int,
        #                     help='In first few epochs, use teahcer forcing to train')
        # parser.add_argument('--tf_ratio', default=0.5, type=float,
        #                     help='After first X epochs of teacher forcing, do student '
        #                          'learning with random masks of 0.5. Maybe anneal it to 0?')
        parser.add_argument('--obj', default='gaussian', type=str, choices=['gaussian', 'l1', 'l2'],
                            help='The objective would be Gaussian likelihood, l1 loss or l2 loss')
        parser.add_argument('--iptw', default=0, type=int,
                            help='do inverse proposensity weighting or not')
        # Indicator loss: is it helpful?
        # parser.add_argument('--reg_alpha', default=0.9, type=float,
        #                     help='If 1, no indicator loss. If 0 no regression loss. '
        #                          'Should be > 0.5.')
        # parser.add_argument('--use_ind_in_test', default=0, type=int,
        #                     help='If 1, use the indicator prediction to roll-out.')
        return parser

    def trainer_args(self):
        return dict(
            gradient_clip_val=1,
            stochastic_weight_avg=True, # Did not test if it improves but seems cool
        )


class HypotensionLRLightning(HypotensionGRULightning):
    def init_model(self):
        self.model = nn.Linear(len(HypotensionDataset.all_cols), 2*HypotensionDataset.len_f)

    def forward(self, x):
        return self.model(x)

    def _step(self, batch):
        x_list = batch['x_list']
        x_pad = pad_sequence(x_list, batch_first=True)
        # ^-- [batch_size, max_len, input dim]

        out = self.model(x_pad)
        # ^-- [batch_size, max_len, out dim]
        return self.cal_loss(out, x_pad)

    @classmethod
    def get_rs_loader(cls, args):
        rs = RandomSearch(hparams=args, seed=args.seed)
        # rs.add_rs_hparams('seed', short_name='s', chose_from=[321])
        rs.add_rs_hparams('seed', short_name='s', gen=lambda hparams: np.random.randint(100))
        rs.add_rs_hparams('lr', chose_from=[2e-4, 5e-4, 1e-3, 2e-3])
        rs.add_rs_hparams('batch_size', short_name='bs', chose_from=[16, 32, 64])
        return rs

    @staticmethod
    def add_model_specific_args(parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser.add_argument('--epochs', type=int, default=100)
        parser.add_argument('--lr', default=1e-3, type=float)
        parser.add_argument('--batch_size', default=16, type=int)
        return parser

    def trainer_args(self):
        return dict(
            stochastic_weight_avg=True, # Did not test if it improves but seems cool
        )
