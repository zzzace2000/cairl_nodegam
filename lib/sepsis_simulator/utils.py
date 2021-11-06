"""
To get the mdp parameters from sepsis simulator

@author: kingsleychang
"""
import numpy as np
import pandas as pd
import torch

from .sepsisSimDiabetes.DataGenerator import DataGenerator
from .sepsisSimDiabetes.MDP import MDP_DICT
from .sepsisSimDiabetes.State import State
from sklearn.model_selection import train_test_split
import platform
from os.path import join as pjoin, exists as pexists
import os
import pickle


def run_policy(policy, N, mdp='linear', return_trajectories=False, seed=None,
               obs_sigmas=0., gamma=0.9, max_num_steps=20):
    ## First, run the optimal policy to get rewards
    if seed is None:
        seed = np.random.randint(0, 1000)
    dg = DataGenerator(seed=seed, mdp=mdp)

    ### first sim data under optimal policy to get range of what is best
    (states, actions, seq_lens, rewards,
         _, init_observs, observs, init_observs_mask,
         observs_mask, action_probs) = dg.simulate(
            policy, N, max_num_steps=max_num_steps,
            policy_idx_type='full', p_diabetes=0.2,
            output_state_idx_type='full', obs_sigmas=obs_sigmas)

    rewards[np.isinf(rewards)] = 0
    gam_t = np.power(gamma, np.arange(max_num_steps))
    returns = np.sum(rewards * gam_t, axis=1)
    avg_returns = np.mean(returns)

    if not return_trajectories:
        return avg_returns

    observs[np.isinf(observs)] = 0 # The val after end time is -inf
    mu = 0.0
    for t in range(observs.shape[1]):
        mu += observs[:, t, :] * (gamma ** t)
    mu_mean = np.mean(mu, axis=0)

    D = {'o_init': init_observs, 'o': observs, 's': states,
         'a': actions, 'len': seq_lens, 'mu': mu_mean, 'r': rewards,
         'seed': seed, 'N': N, 'reward': avg_returns, 'gamma': gamma,
         'max_num_steps': max_num_steps}
    return avg_returns, D


def run_policy_to_get_exp(
        num_exp, policy, mdp='linear', seed=None, obs_sigmas=0.,
        max_num_steps=20):
    the_mdp = MDP_DICT[mdp](
        init_state_idx=None,  # Random initial state
        policy_array=policy, policy_idx_type='full',
        p_diabetes=0.2, seed=seed)

    # Set the default value of states / actions to negative -1,
    iter_obs = np.ones((num_exp, State.PHI_DIM), dtype=np.float32) * (-1)
    iter_actions = np.ones(num_exp, dtype=int) * (-1)
    iter_obs_next = np.ones((num_exp, State.PHI_DIM), dtype=np.float32) * (-1)
    iter_s = np.ones((num_exp), dtype=np.int64) * (-1)
    iter_s_next = np.ones((num_exp), dtype=np.int64) * (-1)

    # Start
    the_mdp.state = the_mdp.get_new_state()
    t = 0
    for i in range(num_exp):
        iter_obs[i] = the_mdp.state.get_phi_vector()
        iter_s[i] = the_mdp.state.get_state_idx(idx_type='full')
        # this_obs = o_init + obs_sigmas * self.rng.normal(0, 1, NUM_OBS)

        step_action = the_mdp.select_actions()  # policy takes action & returns Action object
        iter_actions[i] = step_action.get_action_idx().astype(int)

        # t+1
        step_reward = the_mdp.transition(step_action)
        iter_obs_next[i] = the_mdp.state.get_phi_vector()
        iter_s_next[i] = the_mdp.state.get_state_idx(idx_type='full')

        t += 1
        if t == max_num_steps:
            the_mdp.state = the_mdp.get_new_state()
            t = 0

    return {
        'o': iter_obs,
        'o_next': iter_obs_next,
        'a': iter_actions,
        's': iter_s,
        's_next': iter_s_next,
    }


def train_test_split_D(D, val_ratio=0.2, seed=321):
    '''
    Split the sepsis database into train and val
    '''
    if val_ratio > 0:
        train_D, val_D = {}, {}

        train_D['s'], val_D['s'], \
        train_D['o_init'], val_D['o_init'], \
        train_D['o'], val_D['o'], \
        train_D['r'], val_D['r'], \
        train_D['a'], val_D['a'], \
            = train_test_split(
            D['s'], D['o_init'], D['o'], D['r'], D['a'],
            test_size=val_ratio, random_state=seed, shuffle=True,
        )
        train_D['max_num_steps'] = val_D['max_num_steps'] = D['max_num_steps']
        train_D['gamma'] = val_D['gamma'] = D['gamma']
        val_D['N'] = int(val_ratio * D['N'])
        train_D['N'] = D['N'] - val_D['N']

    return train_D, val_D


def load_mma_model(name):
    ''' Follow the stored location in run_mma.py. Load the model based on val perf '''
    best_path = pjoin('logs', name, 'mma.pkl')
    # My-specific helper function
    is_in_q_server = (platform.node().startswith('vws') or platform.node().startswith('q'))
    if not pexists(best_path) and is_in_q_server:
        cmd = f'rsync -avzL v:/h/kingsley/irl_nodegam/logs/{name} ./logs/'
        print(cmd)
        os.system(cmd)

    assert pexists(best_path), f'No {best_path} exists!'

    with open(best_path, 'rb') as fp:
        params = pickle.load(fp)

    W = params['weight'][np.argmax(params['val_a'])]
    def model(x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()
        elif isinstance(x, pd.DataFrame):
            x = x.values
        return x @ W

    return model
