import argparse
import logging
import os
import pickle
from os.path import exists as pexists

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from lib.sepsis_simulator.feature_expectations import SepsisMCMuEstimator
from lib.mma import MaxMarginAbbeel
from lib.sepsis_simulator.policy import SepsisOptimalSolver
from lib.sepsis_simulator.sepsisSimDiabetes.Action import Action
from lib.sepsis_simulator.sepsisSimDiabetes.State import State
from lib.sepsis_simulator.utils import run_policy, train_test_split_D
from lib.utils import output_csv

# Use it to create figure instead of interactive
matplotlib.use('Agg')


def get_bc_policy(D, smooth_amount=0.5):
    pi_init = np.zeros((State.NUM_FULL_STATES, Action.NUM_ACTIONS_TOTAL)) + smooth_amount
    # Record the first state and first action
    for ps, pa in zip(D['s'], D['a']):
        for s, a in zip(ps[:-1], pa):
            pi_init[s, a] += 1

    # Normalize to get probability
    the_sum = pi_init.sum(axis=-1, keepdims=True)
    # is_zero = (the_sum == 0)
    # the_sum[is_zero] = 1
    pi_init /= the_sum
    assert np.allclose(np.sum(pi_init, axis=1), 1), 'Not sum to 1! Not a policy.'

    # If no count in certain state, just choose random actions
    # pi_init[np.broadcast_to(is_zero, pi_init.shape)] = (1. / Action.NUM_ACTIONS_TOTAL)
    return pi_init


parser = argparse.ArgumentParser(description='run the inverse RL')
parser.add_argument('--name', type=str, default='debug', help='name')
parser.add_argument('--mdp', type=str, choices=['original', 'gam', 'linear'],
                    default='linear', help='How to generate reward.')
parser.add_argument('--fold', type=int, default=0)
parser.add_argument('--seed', type=int, default=321)
parser.add_argument('--gamma', type=float, default=0.9,
                    help='Decay rate in RL. Set it to 0.9 to encourage treating patients '
                         'earlier to leave the hospitals.')
parser.add_argument('--N', type=int, default=5000,
                    help='Number of samples generated')
parser.add_argument('--precision', type=float, default=1e-3,
                    help='The terination precision for IRL optimization.')
parser.add_argument('--opt_method', type=str, default='max_margin',
                    choices=['max_margin', 'projection'],
                    help='Two ways for optimizing apprenticeship learning.')
parser.add_argument('--pi_init', type=str, default='uniform',
                    choices=['uniform', 'bc'],
                    help='How to initialize policy pi. Use uniform or behavior cloning.')
parser.add_argument('--n_iter', type=int, default=60,
                    help='Number of iterations to run MMA IRL')
parser.add_argument('--disc_state_type', type=str, default='next',
                    choices=['next', 'current'], help='use next state or current state')
args = parser.parse_args()


# Set seed
np.random.seed(args.seed)

#### Load data
expert_dest = f"./data/sepsisSimData/" \
              f"{args.mdp}MDP_N{args.N}_g{args.gamma}_f{args.fold}_expert_data.pkl"
assert pexists(expert_dest), 'Run sepsis_expert_gen.py first to generate!'
with open(expert_dest, 'rb') as fp:
    ed = pickle.load(fp)

if args.pi_init == 'uniform':
    pi_init = 1. / Action.NUM_ACTIONS_TOTAL \
              * np.ones((State.NUM_FULL_STATES, Action.NUM_ACTIONS_TOTAL))
elif args.pi_init == 'bc':
    pi_init = get_bc_policy(ed['optimal']['train_D'])
else:
    raise NotImplementedError('No such pi_init exists: ' + args.pi_init)

mdp_solver = SepsisOptimalSolver(discount=args.gamma, mdp=args.mdp)

train_D, val_D = train_test_split_D(ed['optimal']['train_D'], val_ratio=0.2, seed=321)
test_D = ed['optimal']['test_D']

# Evaluate action matched
class ActionMatchedEvaluator(object):
    def __init__(self, split, D):
        self.split = split
        self.D = D

    def evaluate(self, pi, stochastic=True):
        matched, total = 0., 0
        for ps, pa in zip(self.D['s'], self.D['a']):
            for s, a in zip(ps[:-1], pa):
                if not stochastic:
                    matched += int(a == np.argmax(pi[a]))
                else:
                    matched += (pi[s, a]) # The probability
                total += 1

        return {f'{self.split}_a': matched / total}


class RewardEvaluator(object):
    def __init__(self, N, eval_times):
        self.N = N
        self.eval_times = eval_times

    def evaluate(self, pi, stochastic=True):
        # 10k takes around 15s
        all_r = [
            run_policy(pi, N=self.N, gamma=args.gamma, seed=seed, mdp=args.mdp)
            for seed in range(self.eval_times)
        ]
        return {'r': np.mean(all_r), 'r_std': np.std(all_r)}


evaluators = [
    ActionMatchedEvaluator('val', val_D),
    ActionMatchedEvaluator('test', test_D),
    RewardEvaluator(N=20000, eval_times=3),
]
mu_estimator = SepsisMCMuEstimator(
    seed=ed['optimal']['train_D']['seed'], gamma=args.gamma,
    state_type=args.disc_state_type, mdp=args.mdp,
)

mu_expert = ed['optimal']['train_D']['mu']
if args.disc_state_type == 'current':
    o_init = ed['optimal']['train_D']['o_init']
    o = ed['optimal']['train_D']['o']
    gamma = ed['optimal']['train_D']['gamma']

    states = np.concatenate([o_init[:, None, :], o[:, :-1, :]], axis=1)
    states[np.isinf(states)] = 0
    # ^-- [n_sample, T, 5]

    mu_expert = 0.0
    for t in range(states.shape[1]):
        mu_expert += states[:, t, :] * (gamma ** t)
    mu_expert = np.mean(mu_expert, axis=0)

mma = MaxMarginAbbeel(pi_init=pi_init,
                      p=State.PHI_DIM, # dim of phi(s, a)
                      mu_expert=mu_expert,
                      mdp_solver=mdp_solver,
                      evaluators=evaluators,
                      irl_precision=args.precision,
                      method=args.opt_method,
                      mu_estimator=mu_estimator,
                      stochastic=True)

params, eval_metrics = mma.run(n_iteration=args.n_iter)

os.makedirs('./results', exist_ok=True)

ret_results = dict()
ret_results['name'] = args.name
ret_results['mdp'] = args.mdp
ret_results['opt_r'] = ed['optimal_reward_mean']
ret_results['opt_r_std'] = ed['optimal_reward_std']

# Record the behavior cloning performance
pi_bc = get_bc_policy(ed['optimal']['train_D'], smooth_amount=1e-5)
for e in evaluators:
    for k, v in e.evaluate(pi_bc).items():
        ret_results[f'bc_{k}'] = v

# Take the best eval_metrics. Take the best idx from later...
va = eval_metrics['val_a']
ret_results['best_idx'] = len(va) - np.argmax(va[::-1]) - 1
ret_results['best_val_a'] = eval_metrics['val_a'][ret_results['best_idx']]
ret_results['best_test_a'] = eval_metrics['test_a'][ret_results['best_idx']]
ret_results['best_r'] = eval_metrics['r'][ret_results['best_idx']]
ret_results['best_r_std'] = eval_metrics['r_std'][ret_results['best_idx']]
# Also record the performance of pi_best: the weighted sum of policy
if ret_results['best_idx'] == 0:
    ret_results['pi_best_test_a'] = ret_results['pi_best_r'] = -1
else:
    ret_results['pi_best_test_a'] = eval_metrics['best_test_a'][ret_results['best_idx']-1]
    ret_results['pi_best_r'] = eval_metrics['best_r'][ret_results['best_idx']-1]
ret_results.update(vars(args))

output_csv(ret_results, f'./results/mma_new3.csv')

# plot the figure
os.makedirs(f'./results/figures/{args.name}', exist_ok=True)
for k in eval_metrics:
    plt.plot(eval_metrics[k])
    plt.title(k)
    plt.savefig(f'./results/figures/{args.name}/{k}.jpg')
    plt.close()


# Finally save the parameters and eval_metrics
params.update(eval_metrics)
os.makedirs(f'logs/{args.name}/', exist_ok=True)
with open(f'logs/{args.name}/mma.pkl', 'wb') as op:
    pickle.dump(params, op)


logging.info('optimal policy test value: %.3f +- %.3f'
      % (ed['optimal_reward_mean'], ed['optimal_reward_std']))
logging.info('Finish running!')



# from lib.sepsis_simulator.sepsisSimDiabetes.MDP import MDP, GAMMDP, LinearMDP
# def phi(s, a):
#     '''
#     The phi(s, a) used by IRL as basis to recover reward.
#     Here we follow our simulator to generate a reward from the next state.
#
#     s: the state index
#     a: the action index
#         return: the next state vector (5 dim)
#     '''
#     mdp_cls = {'original': MDP, 'gam': GAMMDP, 'linear': LinearMDP}[args.mdp]
#     this_mdp = mdp_cls(init_state_idx=s, init_state_idx_type='full')
#     _ = this_mdp.transition(Action(action_idx=a))
#
#     obs = this_mdp.state.get_phi_vector()
#     return obs