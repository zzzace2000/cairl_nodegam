#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""


Wrapper around the sepsis simulator to get trajectories & optimal policy out.
Behavior for our purposes will be eps-greedy of optimal.

Lots of code here is directly copied from the original gumbel-max-scm repo.s

@author: kingsleychang
"""

# Sepsis Simulator code
import itertools as it
import os
import pickle
from os.path import exists as pexists
from time import time

import numpy as np
from scipy.linalg import block_diag

from .cf import counterfactual as cf
from .sepsisSimDiabetes.Action import Action
from .sepsisSimDiabetes.MDP import MDP_DICT


class SepsisOptimalSolver(object):
    '''
    Get the optimal policy given the transition matrix of MDP
    and reward matrix. Used in getting expert policy and IRL.
    :params tx_mat_full: a matrix of size [n_actions, n_states, n_states]. Each
        entry represents the probability of transtion to next state under
        current state and action.
    :params r_mat_full: a matrix of size [n_actions, n_states, n_states]. Each
        entry represents the reward received under (s, a, s').
    '''
    def __init__(self, mdp='gam', discount=0.99, state_time='next'):
        assert mdp in ['clinear', 'cgam', 'gam', 'linear', 'original', 'cogam']
        self.mdp = mdp
        self.discount = discount
        self.state_time = state_time
        self.state_cls = MDP_DICT[self.mdp].state_cls

        tx_mat_full, r_mat_full = self.get_transition_and_reward_mat()
        self.tx_mat_full = tx_mat_full
        self.r_mat_full = r_mat_full
        # self.num_states = self.state_cls.NUM_FULL_STATES
        # self.num_actions = Action.NUM_ACTIONS_TOTAL
        self.all_next_states = None

    def solve(self, reward_fn=None, batch_size=-1):
        '''
        This solves the optimal policy for sepsis MDP.
        - reward_fn: it takes in the next obs phi(s') and then output the reward
            numpy array of [s, a, s']. If none, use ground truth reward.
        - batch_size: the bs for reward_fn. If -1, it uses all states (1440)
            to pass to reward_fn.
        '''
        if reward_fn is None: # Used the ground truth reward matrix
            r_mat_full = self.r_mat_full
        else: # Used in the Apprenticeship learning
            if self.all_next_states is None:
                self.all_next_states = np.empty(
                    (self.state_cls.NUM_FULL_STATES, self.state_cls.PHI_DIM), dtype=np.float32)

                for i in range(self.state_cls.NUM_FULL_STATES):
                    self.all_next_states[i] = self.state_cls(state_idx=i, idx_type='full')\
                        .get_phi_vector().astype(np.float32)

            # r_mat_full = reward_fn(self.all_next_states)
            r_mat_full = np.empty((Action.NUM_ACTIONS_TOTAL, self.state_cls.NUM_FULL_STATES,
                                   self.state_cls.NUM_FULL_STATES), dtype=np.float32)
            if batch_size == -1:
                batch_size = self.state_cls.NUM_FULL_STATES

            for bi in range(0, self.state_cls.NUM_FULL_STATES, batch_size):
                r = reward_fn(self.all_next_states[bi:(bi + batch_size)])
                if self.state_time == 'next':
                    r_mat_full[:, :, bi:(bi + batch_size)] = r[None, None, :]
                else:
                    r_mat_full[:, bi:(bi + batch_size), :] = r[None, :, None]

        fullMDP = cf.MatrixMDP(self.tx_mat_full, r_mat_full)
        fullPol = fullMDP.policyIteration(discount=self.discount, eval_type=1)
        return fullPol

    def get_transition_and_reward_mat(self):
        mdp_params = self.learn_mdp_params(mdp=self.mdp)

        tx_mat = mdp_params["tx_mat"]
        r_mat = mdp_params["r_mat"]

        tx_mat_full = np.zeros(
            (Action.NUM_ACTIONS_TOTAL, self.state_cls.NUM_FULL_STATES, self.state_cls.NUM_FULL_STATES))
        r_mat_full = np.zeros(
            (Action.NUM_ACTIONS_TOTAL, self.state_cls.NUM_FULL_STATES, self.state_cls.NUM_FULL_STATES))

        for a in range(Action.NUM_ACTIONS_TOTAL):
            tx_mat_full[a, ...] = block_diag(tx_mat[0, a, ...], tx_mat[1, a, ...])
            r_mat_full[a, ...] = block_diag(r_mat[0, a, ...], r_mat[1, a, ...])

        return tx_mat_full, r_mat_full

    def learn_mdp_params(self, mdp='gam'):
        '''
        Get the parameters for the sepsis simulator by drawing a lot of trajs.

        This is used to generate the true MDP parameters, by
        sampling 10k times from the transitions of every state/action
        pair using the underlying simulator. In the paper, this is used
        to construct an "optimal" physician policy against which we
        compare the RL policy.

        Lots of code here is directly copied from the original POPCORN repo:
        https://github.com/dtak/POPCORN-POMDP
        And the file is mostly copied from the original simulator repo:
        https://github.com/clinicalml/gumbel-max-scm/blob/master/learn_mdp_parameters.ipynb

        Here is the description of simulator from the paper:
        Our simulator includes four vital signs
        (heart rate, blood pressure, oxygen concentration, and glucose levels)
        with discrete states (e.g., low, normal, high),
        along with three treatment options (antibiotics, vasopressors,
        and mechanical ventilation), all of which can be applied at
        each time step. Reward is +1 for discharge of a patient,
        and -1 for death. Discharge occurs only when all patient
        vitals are within normal ranges, and all treatments have been
        stopped. Death occurs if at least three of the vital signs are
        simultaneously out of the normal range. In addition,
        a binary variable for diabetes is present with 20% probability,
        which increases the likelihood of fluctuating glucose levels.

        Realize above statements are wrong. In the code, they have
        3,3,2,5,2,2,2 that corresponds to
        [hr_state, sysbp_state, percoxyg_state, glucose_state,
        antibiotic_state, vaso_state, vent_state].
        So total 720 states.
        '''
        # Choose which MDP to generate reward!
        mdp_cls = MDP_DICT[mdp]
        os.makedirs('./data/sepsisSimData/', exist_ok=True)

        ## First load the transition matrix
        prefix = ''
        if mdp in ['clinear', 'cgam']:
            prefix = 'c_'
        if mdp in ['colinear', 'cogam']:
            prefix = 'co_'

        mdp_param_path = f'./data/sepsisSimData/{prefix}txr_mats.pkl'
        if pexists(mdp_param_path):
            with open(mdp_param_path, 'rb') as f:
                mat_dict = pickle.load(f)
        else:
            ## This takes 2 hours!
            print('Generating transition matrix in MDP... Might take 2 hours')

            np.random.seed(1)
            n_iter = 10000
            n_actions = Action.NUM_ACTIONS_TOTAL  # 8
            n_states = self.state_cls.NUM_OBS_STATES  # 720
            n_components = 2  # Diabetes

            states = range(n_states)
            actions = range(n_actions)
            components = [0, 1]

            ## TRANSITION MATRIX
            tx_mat = np.zeros((n_components, n_actions, n_states, n_states))

            # Not used, but a required argument
            dummy_pol = np.ones((n_states, n_actions)) / n_actions

            ###NOTE: takes about 2 hours...
            ct = 0
            total = n_components * n_actions * n_states * n_iter
            start_t = time()
            for (c, s0, a, _) in it.product(components, states, actions, range(n_iter)):
                ct += 1
                if ct % 100000 == 99999:
                    print("finished %d/%d, took %.2f" % (ct, total, time() - start_t))

                this_mdp = mdp_cls(init_state_idx=s0, p_diabetes=c)
                r = this_mdp.transition(Action(action_idx=a))
                s1 = this_mdp.state.get_state_idx()
                tx_mat[c, a, s0, s1] += 1

            est_tx_mat = tx_mat / n_iter
            # Extra normalization: we get the transition probability
            est_tx_mat /= est_tx_mat.sum(axis=-1, keepdims=True)

            ## PRIOR ON INITIAL STATE
            prior_initial_state = np.zeros((n_components, n_states))

            for c in components:
                this_mdp = mdp_cls(p_diabetes=c)
                for _ in range(n_iter):
                    s = this_mdp.get_new_state().get_state_idx()
                    prior_initial_state[c, s] += 1

            prior_initial_state = prior_initial_state / n_iter
            # Extra normalization for probability
            prior_initial_state /= prior_initial_state.sum(axis=-1, keepdims=True)

            prior_mx_components = np.array([0.8, 0.2])

            mat_dict = {"tx_mat": est_tx_mat,
                        "p_initial_state": prior_initial_state,
                        "p_mixture": prior_mx_components}

            with open(mdp_param_path, 'wb') as f:
                pickle.dump(mat_dict, f)

        ## REWARD MATRIX
        reward_path = f'./data/sepsisSimData/{mdp}MDP_reward.pkl'
        if pexists(reward_path):
            with open(reward_path, 'rb') as f:
                est_r_mat = pickle.load(f)
        else:
            n_actions = Action.NUM_ACTIONS_TOTAL  # 8
            n_states = self.state_cls.NUM_OBS_STATES  # 720
            n_components = 2  # Diabetes
            states = range(n_states)

            # Calculate the reward matrix explicitly. Note reward only depends on future state!
            est_r_mat = np.zeros((n_components, n_actions, n_states, n_states))
            for s1 in states:
                this_mdp = mdp_cls(init_state_idx=s1, p_diabetes=1)
                r = this_mdp.calculateReward()
                est_r_mat[:, :, :, s1] = r

            with open(reward_path, 'wb') as f:
                pickle.dump(est_r_mat, f)

        mat_dict['r_mat'] = est_r_mat
        return mat_dict


# class DQNSepsis(DQN):
#     def __init__(self, D):
#         s = D["s"]
#         a = D["a"]
#         #r = D["r"]
#         s_next = D["s_next"]
#         absorb = D["done"]
#         n_sample = s.shape[0]
#
#         D_mat = []
#         for i in range(n_sample):
#             t = [s[i], a[i][0], None, s_next[i], None, absorb[i][0]]
#             D_mat.append(t)
#         D_mat = np.array(D_mat)
#
#         super().__init__(env=None,
#                          D=D_mat,
#                          hiddens=[128, 64],
#                          learning_rate=1e-4,
#                          gamma=0.99,
#                          buffer_size=n_sample,
#                          max_timesteps=5*10**4,
#                          print_freq=5000,
#                          layer_norm=True,
#                          exploration_fraction=0.001,
#                          exploration_final_eps=0.001,
#                          policy_evaluate_freq=1000,
#                          param_noise=True,
#                          grad_norm_clipping=10,
#                          buffer_batch_size=256,
#                          action_list=range(25))
#
#     def solve(self, reward_fn=None, return_policy=True):
#         act = self.train(use_batch=True, reward_fn=reward_fn)
#         return ActWrapper(act=act)
