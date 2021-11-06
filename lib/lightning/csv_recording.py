import pytorch_lightning as pl
import os
from os.path import join as pjoin
from os.path import exists as pexists
from collections import OrderedDict
import pandas as pd
import numpy as np
from argparse import Namespace
import sys
import torch

from ..utils import output_csv


class CSVRecordingCallback(pl.Callback):
    def on_validation_end(self, trainer, pl_module):
        # Avoid outputing anything in the val sanity check
        if pl_module.global_step == 0:
            return

        metrics = trainer.callback_metrics

        csv_dict = OrderedDict()
        csv_dict['epoch'] = pl_module.current_epoch
        csv_dict['global_step'] = pl_module.global_step
        # In the metrics csv, the epoch is lagged by 1. Remove it.
        csv_dict.update({k: v for k, v in metrics.items()
                         if k not in ['epoch', 'global_step']})

        result_f = pjoin('logs', pl_module.hparams.name,
                         'val_results.csv')
        os.makedirs(pjoin('logs', pl_module.hparams.name), exist_ok=True)
        output_csv(csv_dict, result_f)

    def on_test_start(self, trainer, pl_module):
        # Clean up the callback metrics before test; Remove train_loss, val_r etc...
        trainer.callback_metrics = {}

    def on_test_end(self, trainer, pl_module):
        metrics = trainer.callback_metrics

        csv_dict = OrderedDict()
        csv_dict['name'] = pl_module.hparams.name
        # Load the validation record from my own recorded csv
        val_result_path = pjoin('logs', pl_module.hparams.name, 'val_results.csv')
        if pexists(val_result_path):
            df = pd.read_csv(val_result_path)

            best_epoch = int(torch.load(pjoin('logs', pl_module.hparams.name, 'best.ckpt'))['epoch'] - 1)
            best_record = df[df.epoch == best_epoch].iloc[0]
            # best_record = df.sort_values(
            #     trainer.checkpoint_callback.monitor,
            #     ascending=(trainer.checkpoint_callback.mode == 'min')).iloc[0]
            csv_dict.update(best_record.to_dict())

        # In the metrics csv, the epoch is lagged by 1. Remove it.
        csv_dict.update({k: v for k, v in metrics.items()
                         if k not in ['epoch', 'global_step']})
        csv_dict.update({k: v for k, v in pl_module.hparams.items()
                         if k not in ['name', 'epoch', 'global_step']})

        postfix = '_test' if pl_module.hparams.name.startswith('debug') else ''
        fname = pjoin('results', f'{pl_module.__class__.__name__}_results{postfix}.csv')
        output_csv(csv_dict, fname)
