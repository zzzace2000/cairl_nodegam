"""Run script for training agents"""

import argparse
import json
import os
import platform
import shutil
import sys
from os.path import exists as pexists, join as pjoin
from pathlib import Path
import torch

import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping

from lib.lightning.airl import AIRLLightning, AIRL_NODEGAM_Lightning, \
    AIRL_MIMIC3_Lightning, AIRL_MIMIC3_NODEGAM_Lightning, AIRL_MIMIC3_Lightning2, AIRL_MIMIC3_NODEGAM_Lightning2, \
    AIRL_MIMIC3_NODEGAM_Lightning3, AIRL_MIMIC3_FCNN_Lightning, AIRL_FCNN_Lightning
from lib.lightning.bc import BC_MIMIC3_Lightning, BC_MIMIC3_MLP_Lightning
from lib.lightning.csv_recording import CSVRecordingCallback
from lib.lightning.model_checkpoint import MyModelCheckpoint
from pytorch_lightning.utilities import AttributeDict


from lib.lightning.gru import HypotensionGRULightning, HypotensionLRLightning


# Use it to create figure instead of interactive
# matplotlib.use('Agg')

# Detect anomaly
torch.autograd.set_detect_anomaly(True)

# Encounter wierd bug: untimeError: received 0 items of ancdata
# Follow tricks here https://github.com/pytorch/pytorch/issues/973#issuecomment-459398189
# Might need to debug where might cause the memory leak? Or set num_workers=0
torch.multiprocessing.set_sharing_strategy('file_system')


def _main(hparams) -> None:
    """Main Training Method"""
    if pexists(pjoin('logs', hparams.name, 'MY_IS_FINISHED')):
        return

    # create sym link on v server
    if not pexists(pjoin('logs', hparams.name)) \
            and 'SLURM_JOB_ID' in os.environ \
            and pexists('/checkpoint/kingsley/%s' % os.environ['SLURM_JOB_ID']):
        os.symlink('/checkpoint/kingsley/%s' % os.environ['SLURM_JOB_ID'],
                   pjoin('logs', hparams.name))

    # Save hyperparameters
    os.makedirs(pjoin('logs', 'hparams'), exist_ok=True)
    with open(pjoin('logs', 'hparams', hparams.name), 'w') as op:
        json.dump(hparams, op)

    print('------- Hyperparameters -------')
    print(hparams)
    print('')

    pl.seed_everything(hparams.seed)
    if torch.cuda.device_count() > hparams.gpus:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join([str(v) for v in range(0, hparams.gpus)])

    model = eval(hparams.arch)(hparams)

    logger = TensorBoardLogger(
        save_dir='./lightning_logs/',
        name=hparams.name,
    )

    # Create a sym link to point to which one is the best checkpt
    checkpoint_callback = MyModelCheckpoint(
        dirpath=pjoin('logs', hparams.name),
        filename='{epoch}-{%s:.2f}' % model.monitor_metric,
        monitor=model.monitor_metric,
        mode=model.monitor_mode,
        save_top_k=getattr(model, 'save_top_k', 1),
        save_last=True,
        verbose=True,
        notsave_epochs=hparams.notsave_epochs,
    )
    callbacks = [checkpoint_callback, CSVRecordingCallback()]
    if hparams.patience > 0:
        callbacks.append(
            EarlyStopping(monitor=model.monitor_metric, mode=model.monitor_mode, patience=hparams.patience))

    trainer_args = model.trainer_args()

    last_ckpt = pjoin('logs', hparams.name, 'last.ckpt')
    if pexists(last_ckpt):
        trainer_args['resume_from_checkpoint'] = last_ckpt

    trainer = pl.Trainer(
        gpus=hparams.gpus,
        distributed_backend=hparams.backend,
        max_epochs=hparams.epochs,
        profiler='simple',
        callbacks=callbacks,
        logger=logger,
        deterministic=True,
        **trainer_args
    )
    trainer.fit(model)
    trainer.test(ckpt_path=pjoin('logs', hparams.name, 'best.ckpt'))
    Path(pjoin('logs', hparams.name, 'MY_IS_FINISHED')).touch()


def main(hparams) -> None:
    try:
        _main(hparams)
    finally:
        if pexists(pjoin('is_running', hparams.name)):  # release it
            os.remove(pjoin('is_running', hparams.name))


def random_search(rs_loader, random_search, use_slurm=False):
    os.makedirs('is_running', exist_ok=True)

    num_failed = 0
    for rand_idx, hparams in enumerate(rs_loader):
        print(f'Searching for {rand_idx} / {random_search}')
        if num_failed > 50:
            print('NO MORE HPARAMS FOUND! Exit!')
            sys.exit()

        if pexists(pjoin('is_running', hparams.name)):
            num_failed += 1
            continue

        if (not hparams.ignore_prev_runs) and \
                pexists(pjoin('logs', hparams.name, 'MY_IS_FINISHED')):
            num_failed += 1
            continue

        num_failed = 0
        Path(pjoin('is_running', hparams.name)).touch()

        if not use_slurm:
            main(hparams)
        else:
            cmd = './my_sbatch python -u main.py {}'.format(
                " ".join([f'--{k} {" ".join([str(t) for t in v]) if isinstance(v, list) else v}'
                          for k, v in hparams.items()
                          if v is not None and k not in ['random_search', 'ignore_prev_runs']]))
            os.system(cmd)

        if (rand_idx+1) >= random_search:
            break

def load_from_prev_hparams(args):
    hparams = None
    if pexists(pjoin('logs', 'hparams', args.name)):
        with open(pjoin('logs', 'hparams', args.name)) as fp:
            hparams = json.load(fp)
    elif args.load_from_hparams is not None:
        with open(pjoin('logs', 'hparams', args.load_from_hparams)) as fp:
            hparams = json.load(fp)
    return hparams


def load_user_hparams(parser):
    for action in parser._actions:
        action.default = argparse.SUPPRESS
    return AttributeDict(vars(parser.parse_args()))


def update_args(args, parser, prev_hparams):
    user_hparams = load_user_hparams(parser)
    for k, v in prev_hparams.items():
        if k not in user_hparams:
            setattr(args, k, v)


def clean_up(name):
    shutil.rmtree(pjoin('logs', name), ignore_errors=True)
    shutil.rmtree(pjoin('lightning_logs', name), ignore_errors=True)
    if pexists(pjoin('logs', 'hparams', name)):
        os.remove(pjoin('logs', 'hparams', name))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(add_help=False)

    # Specify which experiemnt to run
    archs = [c.__name__ for c in [AIRLLightning, AIRL_NODEGAM_Lightning,
                                  HypotensionGRULightning, HypotensionLRLightning,
                                  AIRL_MIMIC3_Lightning, AIRL_MIMIC3_NODEGAM_Lightning,
                                  AIRL_MIMIC3_Lightning2, AIRL_MIMIC3_NODEGAM_Lightning2,
                                  AIRL_MIMIC3_NODEGAM_Lightning3,
                                  BC_MIMIC3_Lightning, BC_MIMIC3_MLP_Lightning,
                                  AIRL_MIMIC3_FCNN_Lightning, AIRL_FCNN_Lightning]]
    parser.add_argument('--arch', type=str, default='BC_MIMIC3_Lightning',
                        choices=archs)

    parser.add_argument('--name', type=str, default='debug')
    parser.add_argument('--gpus', type=int, default=1)
    parser.add_argument('--backend', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--random_search', type=int, default=0)
    parser.add_argument('--qos', type=str, default='normal')

    parser.add_argument('--ignore_prev_runs', type=int, default=0)
    parser.add_argument('--load_from_hparams', type=str, default=None)

    parser.add_argument('--notsave_epochs', type=int, default=0,
                        help='This is for when training AIRL the first few epochs have '
                             'wierd graph, could be because the bc_kl reg. So only early'
                             'stop after these number of epochs.')

    temp_args, _ = parser.parse_known_args()
    # Remove stuff if in debug mode
    if temp_args.name.startswith('debug'):
        clean_up(temp_args.name)

    # Load previous hparams arch
    prev_hparams = load_from_prev_hparams(temp_args)
    if prev_hparams is not None:
        temp_args.arch = prev_hparams['arch']
    parser = eval(temp_args.arch).add_model_specific_args(parser)
    args = parser.parse_args()

    # If loading previous hparams, update prev hparams with user inputs
    if prev_hparams is not None:
        update_args(args, parser, prev_hparams)

    if args.random_search == 0:
        main(AttributeDict(vars(args)))
        sys.exit()

    rs_loader = eval(temp_args.arch).get_rs_loader(args)
    rs_loader.ignore(load_user_hparams(parser))

    random_search(rs_loader, args.random_search,
                  use_slurm=(not platform.node().startswith('vws')))
