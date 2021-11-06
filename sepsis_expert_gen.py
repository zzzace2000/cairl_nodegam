#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Wrapper around the sepsis simulator to get trajectories & optimal policy out.
Behavior for our purposes will be eps-greedy of optimal.

Lots of code here is directly copied from the original gumbel-max-scm repo.s
and the POPCORN repo

@author: kingsleychang
"""

import argparse
import pickle
import sys
from os.path import exists as pexists

import numpy as np
import os

from lib.sepsis_simulator.sepsisSimDiabetes.Action import Action
from lib.sepsis_simulator.policy import SepsisOptimalSolver
from lib.sepsis_simulator.utils import run_policy


parser = argparse.ArgumentParser(description='generate true sepsis MDP parameters and '
                                             'expert trajectories')
parser.add_argument('--mdp', type=str, choices=['original', 'gam', 'linear', 'cgam', 'clinear'],
                    default='gam', help='How to generate reward.')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--gamma', type=float, default=0.9,
                    help='Decay rate in RL. Set it to 0.9 to encourage treating patients '
                         'earlier to leave the hospitals.')
parser.add_argument('--N', type=int, default=5000,
                    help='Number of samples generated')
args = parser.parse_args()


expert_dest = f"./data/sepsisSimData/" \
              f"{args.mdp}MDP_N{args.N}_g{args.gamma}_f{args.fold}_expert_data.pkl"
# if pexists(expert_dest):
#     print('Already finish running expert data! Exit.')
#     sys.exit()


# Set the epsilon based on my tradeoff parameters in MaxEnt framework
MAX_NUM_STEPS = 20
phys_epsilons = [0.07, 0.14]

solver = SepsisOptimalSolver(mdp=args.mdp, discount=args.gamma)
optimal_policy = solver.solve()

# Training data
def run(eps):
    eps_greedy_policy = np.copy(optimal_policy)
    eps_greedy_policy[eps_greedy_policy == 1] = 1 - eps
    eps_greedy_policy[eps_greedy_policy == 0] = eps / (Action.NUM_ACTIONS_TOTAL - 1)

    avg_eps_returns, train_D = \
        run_policy(eps_greedy_policy, mdp=args.mdp, N=args.N, gamma=args.gamma,
                   seed=(args.fold + 1), return_trajectories=True)
    print(f'eps{eps} policy train value: %.3f' % avg_eps_returns)

    # Test data
    test_avg_eps_returns, test_D = \
        run_policy(eps_greedy_policy, mdp=args.mdp, N=args.N, gamma=args.gamma,
                   seed=(args.fold + 100), return_trajectories=True)
    print(f'eps{eps} policy test value: %.3f' % test_avg_eps_returns)
    return train_D, test_D


save_dict = dict()
for eps in phys_epsilons:
    train_D, test_D = run(eps)

    save_dict[f'eps{eps}'] = {}
    save_dict[f'eps{eps}']['train_D'] = train_D
    save_dict[f'eps{eps}']['test_D'] = test_D

save_dict[f'optimal'] = {}
train_D, test_D = run(eps=0)
save_dict[f'optimal']['train_D'] = train_D
save_dict[f'optimal']['test_D'] = test_D


# optimal reward
opt_returns = []
for seed in range(10):
    avg_opt_returns = run_policy(
        optimal_policy, mdp=args.mdp, N=10000, gamma=args.gamma, seed=seed)
    opt_returns.append(avg_opt_returns)

save_dict['optimal_reward_mean'] = np.mean(opt_returns)
save_dict['optimal_reward_std'] = np.std(opt_returns)

print('optimal policy test value: %.3f +- %.3f'
      % (save_dict['optimal_reward_mean'], save_dict['optimal_reward_std']))

# Save the expert demonstrations
os.makedirs('./data/sepsisSimData/', exist_ok=True)
with open(expert_dest, 'wb') as op:
    pickle.dump(save_dict, op)
