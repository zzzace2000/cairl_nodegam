import torch

import os
import pickle
from os.path import exists as pexists
from torch.utils.data import Dataset
import numpy as np
from .utils import train_test_split_D


class SepsisExpertDataset(Dataset):
    def __init__(self, mdp, N, gamma, fold=0, split='train', val_ratio=0.2,
                 expert_pol='optimal'):
        assert split in ['train', 'val', 'test'], f'Wrong split: {split}'

        self.expert_pol = expert_pol
        expert_dest = f"./data/sepsisSimData/" \
                      f"{mdp}MDP_N{N}_g{gamma}_f{fold}_expert_data.pkl"
        assert pexists(expert_dest), 'Run sepsis_expert_gen.py first to generate!'
        with open(expert_dest, 'rb') as fp:
            ed = pickle.load(fp)

        if split == 'test':
            D = ed[expert_pol]['test_D']
        else:
            train_D, val_D = train_test_split_D(
                ed[expert_pol]['train_D'], val_ratio=val_ratio, seed=321)
            D = train_D if split == 'train' else val_D

        self.experiences = []
        # Generate per-time experience
        all_obs = np.concatenate([D['o_init'][:, None], D['o']], axis=1).astype(np.float32)
        D['r'] = D['r'].astype(np.float32)
        for idx in range(D['N']):
            for t in range(D['max_num_steps']):
                exp = dict()
                exp['s'] = D['s'][idx, t]
                exp['o'] = all_obs[idx, t]
                exp['a'] = D['a'][idx, t]
                exp['r'] = D['r'][idx, t]
                exp['o_next'] = all_obs[idx, t + 1]
                if t == (D['max_num_steps'] - 1):
                    exp['done'] = True
                else:
                    exp['done'] = False
                self.experiences.append(exp)

    def __len__(self):
        return len(self.experiences)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        return self.experiences[idx]
