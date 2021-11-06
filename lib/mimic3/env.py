import torch
import torch.nn as nn
from os.path import join as pjoin, exists as pexists
import numpy as np
from ..lightning.utils import load_best_model_from_trained_dir
from .dataset import HypotensionDataset

'''
- Need expert data initial states
- Need simulator
- Need policy
'''


class MIMIC3HypotensionEnv(nn.Module):
    '''
    Potential TODO:
        (1) Add the patient length into it?
        (2) When taking same action, returned the observed next state?
            - Problem: non-markov states or hidden confounder?
    '''
    def __init__(self, data_dir, ts_model_name, policy, split='train', max_time=36):
        super().__init__()
        self.data_dir = data_dir
        self.ts_model_name = ts_model_name
        self.policy = policy
        self.max_time = max_time

        # Load the initial states
        # self.init_states = torch.load(pjoin(data_dir, f'{split}_init_states.pkl'))
        self.ts_model = load_best_model_from_trained_dir(ts_model_name)

    def run_policy_to_get_exp(self, num_exp, policy, batch_size):
        '''
        num_exp: generate number of experience
        '''
        with torch.no_grad():
            return self._run_policy_to_get_exp(num_exp, policy, batch_size)

    def _run_policy_to_get_exp(self, num_exp, policy, batch_size):
        '''
        num_exp: generate number of experience
        '''
        total_num = int(np.ceil(num_exp / self.max_time))

        # Randomly sample the initial states
        idxes = []
        for _ in range(int(np.ceil(total_num / len(self.init_states)))):
            idx = torch.randperm(self.init_states.shape[0])
            idxes.append(idx)
        idxes = torch.cat(idxes)[:total_num]

        device = next(self.ts_model.parameters()).device
        results = []
        for s in range(0, total_num, batch_size): # Mini-batch of patients
            states = self.init_states[idxes[s:(s+batch_size)]].to(device).unsqueeze(1)
            # ^-- [B, 1, D]

            hiddens = None
            for t in range(1, HypotensionDataset.MAX_TIME):
                actions = policy.sample_action(states[:, -1, :])

                next_state, hiddens = self.ts_model.pred_next(states, actions, hx=hiddens)
                states = torch.cat([states[:, :-1, :], last_states_act, next_state], dim=1)
            results.append(states)
        results = torch.cat(results, dim=0)
        return results
