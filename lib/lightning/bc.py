#!/usr/bin/env python3
# coding=utf-8
"""
A potential tutorial for GRUCell
https://towardsdatascience.com/encoder-decoder-model-for-multistep-time-series-forecasting-using-pytorch-5d54c6af6e60
"""

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from ..mimic3.dataset import HypotensionDataset
from ..random_search import RandomSearch
from .utils import create_fc_layer


class BC_MIMIC3_Lightning(pl.LightningModule):
    ''' Use GRU to do behavior cloning '''
    monitor_metric = 'val_acc'
    monitor_mode = 'max'

    def __init__(self, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.init_model()

    def init_model(self):
        self.gru = nn.GRU(
            len(HypotensionDataset.all_cols),
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
            n_out=(HypotensionDataset.NUM_VASO_BINS * HypotensionDataset.NUM_FLUID_BINS),
            dropout=self.hparams.fc_dropout,
            in_bn=True,
            is_3d_input=True,
        )

    def forward(self, x_pad, x_len=None):
        x_in = x_pad
        if x_len is not None:
            x_in = pack_padded_sequence(x_pad, x_len, enforce_sorted=False, batch_first=True)
        out, hiddens = self.gru(x_in)
        if x_len is not None:
            out, _ = pad_packed_sequence(out, batch_first=True)

        # ^-- [batch_size, max_len, hidden dim]
        pred = self.out(out)
        return pred

    def training_step(self, batch, batch_idx, hiddens=None):
        ret_dict = self._step(batch, hiddens, is_training=True)

        avg_loss = (ret_dict['sum_loss'] / ret_dict['num'])
        avg_acc = (ret_dict['sum_acc'] / ret_dict['num'])
        logs = {'train_loss': avg_loss, 'train_acc': avg_acc}
        self.log_dict(logs, prog_bar=True)

        result = {'loss': avg_loss, 'log': logs}
        return result

    def validation_step(self, batch, batch_nb):
        return self._step(batch)

    def test_step(self, batch, batch_nb):
        return self._step(batch)

    def _step(self, batch, hiddens=None, is_training=False):
        x_list = batch['x_list']

        x_len = [v.size(0) for v in x_list]
        x_pad = pad_sequence(x_list, batch_first=True)

        # Get current action in each x
        cur_act = HypotensionDataset.extract_cur_a(x_pad, form='act_idx')

        pred = self(x_pad, x_len)
        # ^-- [batch_size, max_len, action dim (16)]
        pred = pred[:, :-1, :] # Remove last time point

        # When calculating 3-d CE loss, the dim=1 is the num of classes
        loss = F.cross_entropy(pred.permute(0, 2, 1), cur_act, reduction='none')
        mask = torch.zeros_like(loss)
        for i, l in enumerate(x_len):
            mask[i, :l] = 1.
        acc = torch.sum((torch.argmax(pred, dim=-1) == cur_act).float() * mask)

        ret_dict = dict()
        ret_dict['sum_loss'] = (loss * mask).sum()
        ret_dict['sum_acc'] = acc
        ret_dict['num'] = mask.sum()
        return ret_dict

    def validation_epoch_end(self, outputs):
        return self._epoch_end(outputs, prefix='val')

    def test_epoch_end(self, outputs):
        return self._epoch_end(outputs, prefix='test')

    def _epoch_end(self, outputs, prefix='val'):
        all_num = torch.stack([x[f'num'] for x in outputs])

        def get_loss(name='loss'):
            all_loss = torch.stack([x[f'sum_{name}'] for x in outputs])
            avg_loss = torch.sum(all_loss) / torch.sum(all_num)
            return avg_loss

        avg_loss = get_loss('loss')
        avg_acc = get_loss('acc')

        tensorboard_logs = {f'{prefix}_loss': avg_loss, f'{prefix}_acc': avg_acc}
        self.log_dict(tensorboard_logs)

        result = {'log': tensorboard_logs}
        result.update(tensorboard_logs)
        return result

    def configure_optimizers(self):  # REQUIRED
        return torch.optim.Adam(self.parameters(), lr=self.hparams.lr, weight_decay=self.hparams.wd)

    def train_dataloader(self):
        return HypotensionDataset.make_loader(
            data_kwargs=dict(
                fold=self.hparams.fold,
                preprocess='quantile',
            ),
            split='train',
            batch_size=self.hparams.batch_size,
            shuffle=True,
            num_workers=0,
            debug=self.hparams.name.startswith('debug'),
        )

    def val_dataloader(self):
        return HypotensionDataset.make_loader(
            data_kwargs=dict(
                fold=self.hparams.fold,
                preprocess='quantile',
            ),
            split='val',
            batch_size=2 * self.hparams.batch_size,
            shuffle=False,
            num_workers=0,
            debug=self.hparams.name.startswith('debug'),
        )

    def test_dataloader(self):
        return HypotensionDataset.make_loader(
            data_kwargs=dict(
                fold=self.hparams.fold,
                preprocess='quantile',
            ),
            split='test',
            batch_size=2 * self.hparams.batch_size,
            shuffle=False,
            num_workers=0,
            debug=self.hparams.name.startswith('debug'),
        )

    @classmethod
    def get_rs_loader(cls, args):
        rs = RandomSearch(hparams=args, seed=args.seed)
        # rs.add_rs_hparams('seed', short_name='s', chose_from=[321])
        rs.add_rs_hparams('seed', short_name='s', gen=lambda hparams: rs.np_gen.randint(200))
        rs.add_rs_hparams('lr', chose_from=[5e-4, 1e-3, 2e-3])
        rs.add_rs_hparams('wd', short_name='wd', chose_from=[0., 1e-6, 1e-5, 1e-4])
        rs.add_rs_hparams('batch_size', short_name='bs', chose_from=[64, 128, 256])
        rs.add_rs_hparams('n_hidden', short_name='nh', chose_from=[64, 128, 256])
        rs.add_rs_hparams('n_layer', short_name='nl', chose_from=[1])
        rs.add_rs_hparams('dropout', short_name='dr',
                          gen=lambda hparams: 0. if hparams.n_layer <= 1 else rs.np_gen.choice([0.3, 0.5]))
        rs.add_rs_hparams('fc_n_hidden', short_name='fnh', chose_from=[128, 256, 384, 512])
        rs.add_rs_hparams('fc_n_layer', short_name='fnl', chose_from=[2, 3, 4])
        rs.add_rs_hparams('fc_dropout', short_name='fdr', chose_from=[0.15, 0.3, 0.5])
        return rs

    @staticmethod
    def add_model_specific_args(parser):
        """
        Specify the hyperparams for this LightningModule
        """
        parser.add_argument('--fold', type=int, default=0)
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
        return parser

    def trainer_args(self):
        return dict(
            gradient_clip_val=1,
            stochastic_weight_avg=True, # Did not test if it improves but seems cool
        )


class BC_MIMIC3_MLP_Lightning(BC_MIMIC3_Lightning):
    ''' Use MLP to do behavior cloning. Test if only seeing states, what's the perf? '''
    def init_model(self):
        # Arch for MLP: with BN, Dropout and ELU...
        self.out = create_fc_layer(
            n_in=len(HypotensionDataset.all_cols),
            n_layer=self.hparams.fc_n_layer,
            n_hidden=self.hparams.fc_n_hidden,
            n_out=(HypotensionDataset.NUM_VASO_BINS * HypotensionDataset.NUM_FLUID_BINS),
            dropout=self.hparams.fc_dropout,
        )

    def forward(self, x):
        return self.out(x)

    def _step(self, batch, hiddens=None, is_training=False):
        # Get current action in each x
        s, a, _, _ = HypotensionDataset.extract_s_and_a_pairs(batch, state_type='all')

        logit = self.out(s)
        loss = F.cross_entropy(logit, a, reduction='sum')

        acc = torch.sum((torch.argmax(logit, dim=-1) == a).float())

        ret_dict = dict()
        ret_dict['sum_loss'] = loss
        ret_dict['sum_acc'] = acc
        ret_dict['num'] = acc.new_tensor(s.shape[0])
        return ret_dict

    @classmethod
    def get_rs_loader(cls, args):
        rs = RandomSearch(hparams=args, seed=args.seed)
        # rs.add_rs_hparams('seed', short_name='s', chose_from=[321])
        rs.add_rs_hparams('seed', short_name='s', gen=lambda hparams: rs.np_gen.randint(200))
        rs.add_rs_hparams('lr', chose_from=[5e-4, 1e-3, 2e-3])
        rs.add_rs_hparams('wd', short_name='wd', chose_from=[0., 1e-6, 1e-5, 1e-4])
        rs.add_rs_hparams('batch_size', short_name='bs', chose_from=[64, 128, 256])
        rs.add_rs_hparams('fc_n_hidden', short_name='fnh', chose_from=[128, 256, 384, 512])
        rs.add_rs_hparams('fc_n_layer', short_name='fnl', chose_from=[2, 3, 4])
        rs.add_rs_hparams('fc_dropout', short_name='fdr', chose_from=[0.15, 0.3, 0.5])
        return rs

    @staticmethod
    def add_model_specific_args(parser):
        """
        Specify the hyperparams for this LightningModule
        """
        # MODEL specific
        parser.add_argument('--epochs', type=int, default=300)
        parser.add_argument('--patience', type=int, default=50)
        parser.add_argument('--lr', default=1e-3, type=float)
        parser.add_argument('--wd', default=0., type=float)
        parser.add_argument('--batch_size', default=32, type=int)
        parser.add_argument('--fc_n_hidden', default=16, type=int)
        parser.add_argument('--fc_n_layer', default=2, type=int)
        parser.add_argument('--fc_dropout', default=0.1, type=float)
        return parser

    def trainer_args(self):
        return dict(
            stochastic_weight_avg=True, # Did not test if it improves but seems cool
        )
