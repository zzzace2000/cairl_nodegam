from os.path import exists as pexists, join as pjoin
import torch.nn as nn
import json
import os
from pytorch_lightning.utilities import AttributeDict
import torch
import numpy as np
import platform
from contextlib import contextmanager


def load_best_model_from_trained_dir(name, target='best'):
    ''' Follow the model architecture in main.py '''
    hparams = load_hparams(name)

    best_path = pjoin('logs', name, '%s.ckpt' % target)
    is_in_q_server = (platform.node().startswith('vws') or platform.node().startswith('q'))
    if not pexists(best_path) and is_in_q_server:
        cmd = f'rsync -avzL v:/h/kingsley/irl_nodegam/logs/{name} ./logs/'
        print(cmd)
        os.system(cmd)

    assert pexists(best_path), f'No {target}.ckpt exists!'

    from .airl import AIRLLightning, AIRL_NODEGAM_Lightning, \
        AIRL_MIMIC3_NODEGAM_Lightning, AIRL_MIMIC3_Lightning, \
        AIRL_MIMIC3_NODEGAM_Lightning2, AIRL_MIMIC3_NODEGAM_Lightning3, \
        AIRL_MIMIC3_Lightning2
    from .gru import HypotensionGRULightning, HypotensionLRLightning
    from .bc import BC_MIMIC3_Lightning, BC_MIMIC3_MLP_Lightning
    model = eval(hparams.arch).load_from_checkpoint(best_path)
    model.train(False)
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_hparams(name):
    os.makedirs(pjoin('logs', 'hparams'), exist_ok=True)
    is_in_q_server = (platform.node().startswith('vws') or platform.node().startswith('q'))
    if not pexists(pjoin('logs', 'hparams', name)) and is_in_q_server:
        cmd = f'rsync -avzL v:/h/kingsley/irl_nodegam/logs/hparams/{name} ./logs/hparams/'
        print(cmd)
        os.system(cmd)

    assert pexists(pjoin('logs', 'hparams', name)), \
        'No hparams exists in %s' % pjoin('logs', 'hparams', name)
    hparams = json.load(open(pjoin('logs', 'hparams', name)))
    hparams = AttributeDict(**hparams)
    return hparams


class Squeeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        return x.squeeze(dim=self.dim)


class MinMaxNormalize(nn.Module):
    ''' Normalize max to +1 and min to -1 '''
    def __init__(self, min=-1, max=1):
        super().__init__()
        if isinstance(min, list) or isinstance(min, np.ndarray):
            min = torch.tensor(min)
            max = torch.tensor(max)

        self.min = nn.Parameter(min, requires_grad=False)
        self.max = nn.Parameter(max, requires_grad=False)

    def forward(self, x):
        return ((x - self.min) / (self.max - self.min) - 0.5) * 2


class Swapaxes(nn.Module):
    def __init__(self, *axes):
        super().__init__()
        self.axes = axes

    def forward(self, x):
        return torch.transpose(x, *self.axes)


def create_fc_layer(n_in, n_layer, n_hidden, n_out, dropout=0.,
                    in_bn=False, is_3d_input=False, use_bn=True):
    def create_bn(n_hidden):
        if not is_3d_input:
            return [nn.BatchNorm1d(n_hidden)]
        return [
            Swapaxes(-1, -2),
            nn.BatchNorm1d(n_hidden),  # Handle 3d normalization but in 1st dim
            Swapaxes(-1, -2),
        ]

    out_layers = []
    if in_bn and use_bn:
        out_layers.extend(create_bn(n_in))

    for i in range(n_layer):
        layer_in = n_in if i == 0 else n_hidden
        layer_out = n_hidden if (i < n_layer - 1) else n_out

        out_layers.append(nn.Linear(layer_in, layer_out))
        if i < (n_layer - 1):
            out_layers.append(nn.ELU())
            if use_bn:
                out_layers.extend(create_bn(layer_out))
            if dropout > 0.:
                out_layers.append(nn.Dropout(p=dropout))
    return torch.nn.Sequential(*out_layers)


@contextmanager
def evaluating(net):
    '''Temporarily switch to evaluation mode.'''
    istrain = net.training
    try:
        net.eval()
        yield net
    finally:
        if istrain:
            net.train()