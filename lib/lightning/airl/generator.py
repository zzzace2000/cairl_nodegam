import argparse

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence, pack_padded_sequence, pad_packed_sequence

from lib.sepsis_simulator.utils import run_policy
from lib.random_search import RandomSearch
# from .common import wrappers
from lib.sepsis_simulator.policy import SepsisOptimalSolver
from lib.sepsis_simulator.sepsisSimDiabetes.Action import Action
from lib.sepsis_simulator.sepsisSimDiabetes.State import State
from lib.sepsis_simulator.utils import run_policy_to_get_exp
from lib.utils import Timer
import torch.nn as nn
from os.path import join as pjoin, exists as pexists
from lib.lightning.utils import load_best_model_from_trained_dir
from lib.mimic3.dataset import HypotensionDataset
from torch.distributions.categorical import Categorical
from lib.lightning.utils import create_fc_layer
from torch.nn import functional as F
from ..utils import evaluating


class SepsisGenerator(object):
    """ Implement Adversarial Imitation RL for MIMIC3 hypotension
    """
    def __init__(self, hparams) -> None:
        super().__init__()
        self.hparams = hparams
        self.mdp_solver = SepsisOptimalSolver(
            discount=self.hparams.model_gamma, mdp=self.hparams.mdp, state_time=self.hparams.disc_state_time,
        )
        # Initial pi is set to uniform
        self.pi = 1. / Action.NUM_ACTIONS_TOTAL \
                  * np.ones((State.NUM_FULL_STATES, Action.NUM_ACTIONS_TOTAL))

        # Initialize a dummy optimizer for code compatibility
        self.dummy_params = torch.nn.Parameter(torch.tensor(1.), requires_grad=False)
        self.optimizer = torch.optim.SGD([self.dummy_params], lr=0.)

    def update(self, reward_disc, step, update=False, **kwargs):
        if (step + 1) % self.hparams.update_gen != 0 and not update:
            return

        def reward_fn(obs_next):
            '''
            Defines the reward function for AIRL.
            - obs_next: the observation (phi) for the next state. Numpy array of
                [State.NUM_FULL_STATES (1440), phi_dim (5)].
            :return r_mat of [num_states, num_actions, num_states]
            '''
            assert obs_next.shape[1] == State.PHI_DIM

            device = 'cuda' if self.hparams.gpus > 0 else 'cpu'
            # Run whole batch. Modify to minibatch if not fit GPU memory
            X = torch.tensor(obs_next, device=device)

            reward = reward_disc(X)
            return reward.cpu().numpy()

        with torch.no_grad(), evaluating(reward_disc):
            self.pi = self.mdp_solver.solve(reward_fn=reward_fn)

    def gen_exp(self, expert_batch):
        num_exp = expert_batch['o_next'].shape[0]

        g_data = run_policy_to_get_exp(
            num_exp=num_exp, policy=self.pi, mdp=self.hparams.mdp)
        return g_data

    def get_action_probs(self, s, a):
        '''
        s: numpy int array [B]
        a: numpy int array [B]
        '''
        if torch.is_tensor(s):
            s = s.cpu().numpy()
        if torch.is_tensor(a):
            a = a.cpu().numpy()
        return self.pi[s, a]

    def eval_step(self, batch, batch_idx):
        probs = self.get_action_probs(batch['s'], batch['a'])
        return {f'a': np.sum(probs), f'total': len(batch['s'])}

    def eval_epoch_end(self, outputs, prefix='val'):
        """Log the avg of the test results"""
        a = sum([x['a'] for x in outputs])
        total = sum([x['total'] for x in outputs])
        avg_a = a / total

        if prefix == 'val':
            eval_N = 5000
            eval_times = 1
        else: # 'test'
            eval_N = 20000
            eval_times = 3

        tensorboard_logs = {f'{prefix}_a': avg_a}
        tensorboard_logs.update({
            f'{prefix}_{k}': v for k, v in self.evaluate_reward(eval_N, eval_times).items()
        })

        return tensorboard_logs

    def evaluate_reward(self, eval_N, eval_times, mdp=None):
        if mdp is None:
            mdp = self.hparams.mdp

        with Timer('Estimating reward', remove_start_msg=False):
            all_rewards = [
                run_policy(self.pi, N=eval_N, gamma=self.hparams.gamma, seed=int(seed), mdp=mdp)
                for seed in range(eval_times)
            ]
            return {
                'reward': np.mean(all_rewards),
                'reward_std': np.std(all_rewards),
            }

    @classmethod
    def get_rs_loader(cls, args, rs=None):
        if rs is None:
            rs = RandomSearch(hparams=args, seed=args.seed)
        rs.add_rs_hparams('update_gen', short_name='ug', chose_from=[20])
        return rs

    @classmethod
    def add_model_specific_args(cls, parser) -> argparse.ArgumentParser:
        """
        Adds arguments for DQN model
        Note: these params are fine tuned for Pong env
        Args:
            parent
        """
        # Optimization
        parser.add_argument('--update_gen', type=int, default=50)
        return parser


class MIMIC3Generator(nn.Module):
    """ Implement a Double DQN here

    Q: how to initalize the policy to be behavior cloning in DQN?

    Q: since in DQN, we can use expert transitions only!
    """
    def __init__(self, hparams) -> None:
        super().__init__()
        self.hparams = hparams

        self.ts_model = load_best_model_from_trained_dir(self.hparams.ts_model_name)

        # Initialize the DQN
        self.build()
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.hparams.gen_lr,
                                          betas=(0.5, 0.99), weight_decay=self.hparams.gen_wd)

        # Cache the gen_exp for disc to speed up.
        self.gen_batch = None

    def build(self):
        def create():
            use_bn = self.hparams.get('use_bn', True)
            return create_fc_layer(
                n_in=len(HypotensionDataset.all_cols),
                n_layer=self.hparams.gen_n_layer,
                n_hidden=self.hparams.gen_n_hidden,
                n_out=HypotensionDataset.TOTAL_ACTIONS,
                dropout=self.hparams.gen_dropout,
                use_bn=use_bn,
            )
        self.net = create()
        self.target_net = create()

    def get_action_probs(self, s, a):
        '''
        s: numpy int array [B]
        a: numpy int array [B]
        '''
        log_prob = self.get_action_log_probs(s, a)
        return torch.exp(log_prob)

    def get_action_log_probs(self, s, a=None, qnet='source'):
        '''
        s: numpy int array [B]
        a: numpy int array [B]
        '''
        assert len(s.shape) == 2

        net = self.net
        if qnet == 'target':
            net = self.target_net

        q_val = net(s) / self.hparams.beta
        log_probs = q_val.log_softmax(dim=1)
        if a is None:
            return log_probs
        return log_probs.gather(1, a.unsqueeze(-1)).squeeze(-1)

    def update(self, reward_disc, step, batch, epoch):
        x_list = batch['x_list']
        if (step + 1) % self.hparams.update_gen != 0:
            return
        # Soft update of target network
        if step % self.hparams.sync_rate == 0:
            self.target_net.load_state_dict(self.net.state_dict())

        # Expert x_list
        states, actions, next_states, dones = HypotensionDataset.extract_s_and_a_pairs(
            x_list, state_type='all')

        if self.hparams.update_q_by_gen > 0 and self.gen_batch is not None:
            # Since we first update disc, we have the gen_exp in disc. Use that to speed up
            gen_states, gen_actions, gen_next_states, gen_dones = self.gen_batch
            states = torch.cat([states, gen_states], dim=0)
            actions = torch.cat([actions, gen_actions], dim=0)
            next_states = torch.cat([next_states, gen_next_states], dim=0)
            dones = torch.cat([dones, gen_dones], dim=0)
            self.gen_batch = None

        # Run DQN to get cur and next q vals
        all_q_vals = self.net(states)
        q_vals = all_q_vals.gather(1, actions.unsqueeze(-1)).squeeze(-1)
        with torch.no_grad(), evaluating(self.target_net):
            # (1) We can take max as normal DQN
            # next_q_vals = self.target_net(next_states).max(1)[0]
            # (2) Or we do sql https://julien-vitay.net/deeprl/EntropyRL.html
            loga = self.get_action_log_probs(next_states, qnet='target')
            next_q = torch.sum(loga.exp() * (self.target_net(next_states) - self.hparams.ent_reg * loga), dim=1)
            next_q[dones] = 0.0
            next_q = next_q.detach()

        # Generate reward by the discriminator (the shaped reward term)
        # Add noise to inputs (at least in GAN lit it improves)
        def add_noise(x):
            if self.hparams.noise == 0 or self.hparams.noise_epochs == 0:
                return x
            n_ratio = max(1 - (epoch / (self.hparams.noise_epochs * self.hparams.epochs)), 0)
            return x + n_ratio * self.hparams.noise * torch.randn_like(x)
        # states, next_states = add_noise(states), add_noise(next_states)

        with evaluating(reward_disc), torch.no_grad():
            # the_s = next_states if self.hparams.disc_state_time == 'next' else states
            # rewards = reward_disc(the_s) # Remove the wierd AIRL term
            # Remove AIRL term but keeps the shaping reward...?
            rewards = reward_disc.get_reward(states, next_states, dones, shaping=False)

        expected_q_vals = rewards + self.hparams.gamma * next_q

        loss = nn.SmoothL1Loss(reduction='none')(q_vals, expected_q_vals)
        if 0 < self.hparams.update_q_by_gen < 1 and self.gen_batch is not None:
            loss[(loss.shape[0] // 2):] *= self.hparams.update_q_by_gen
        loss = torch.mean(loss)

        # Linearly annealing KL loss
        kl_coeff = 1.
        if self.hparams.bc_kl_anneal > 0.:
            kl_coeff = max(1. - (epoch / (self.hparams.bc_kl_anneal * self.hparams.epochs + 1e-3)), 0.)

        if self.hparams.bc_kl > 0 and kl_coeff > 0.:
            # Take the expert state's q value
            exp_q_vals = all_q_vals
            if self.hparams.update_q_by_gen > 0 and self.gen_batch is not None:
                exp_q_vals = exp_q_vals[:(exp_q_vals.shape[0] // 2)]

            bc_prob = torch.cat([p[:-1] for p in batch['bc_prob']], dim=0)

            assert exp_q_vals.shape == bc_prob.shape, f'{exp_q_vals.shape} != {bc_prob.shape}'
            logprob = F.log_softmax(exp_q_vals / self.hparams.beta, dim=-1)
            kl_loss = -torch.mean(torch.sum(bc_prob * logprob, dim=-1))
            kl_loss *= (kl_coeff * self.hparams.bc_kl)

            loss += kl_loss

        # Report current entropy
        with torch.no_grad():
            loga = F.log_softmax(all_q_vals / self.hparams.beta, dim=-1)
            entropy = -torch.mean(torch.sum(torch.exp(loga) * loga, dim=1))
        return {'loss': loss, 'entropy': entropy}

    def gen_exp(self, expert_batch, actions='agent', state_type='features', same_act_use_exp=1.):
        states, actions, new_states, dones = self._gen_exp(
            expert_batch, actions=actions, same_act_use_exp=same_act_use_exp)

        states = HypotensionDataset.extract_s_by_state_type(states, state_type)
        new_states = HypotensionDataset.extract_s_by_state_type(new_states, state_type)
        return states, actions, new_states, dones

    def _gen_exp(self, expert_batch, actions='agent', same_act_use_exp=1.):
        assert actions in ['agent', 'expert']

        x_list = expert_batch['x_list']
        if actions == 'expert' and same_act_use_exp == 1.:
            return HypotensionDataset.extract_s_and_a_pairs(x_list, 'all')

        x_len = [v.size(0) for v in x_list]
        x_pad = pad_sequence(x_list, batch_first=True)
        generated_exp = self.ts_model.simulate(
            x_pad,
            actions_fn=self.sample_action if actions == 'agent' else None,
            sample_stdev=self.hparams.sample_stdev,
            rollout=self.hparams.rollout)
        generated_exp = [ge[:the_len] for ge, the_len in zip(generated_exp, x_len)]

        states, actions, new_states, dones = HypotensionDataset.extract_s_and_a_pairs(generated_exp, 'all')
        if self.hparams.rollout:
            self.gen_batch = states, actions, new_states, dones
            return states, actions, new_states, dones

        # Only do one-step ahead prediction
        exp_states, exp_actions, exp_new_states, exp_dones = \
            HypotensionDataset.extract_s_and_a_pairs(x_list, 'all')

        if same_act_use_exp > 0.:
            is_same = (actions == exp_actions).all(dim=-1)
            new_states[is_same] = (1. - same_act_use_exp) * new_states[is_same] \
                                  + same_act_use_exp * exp_new_states[is_same]

        if actions == 'agent': # Cache
            self.gen_batch = exp_states, actions, new_states, dones
        return exp_states, actions, new_states, dones

    def sample_action(self, s):
        '''
        Sample actions from the DQN.

        states: pytorch array of [B, D]
        '''
        q_val = self.net(s) / self.hparams.beta
        dist = Categorical(logits=q_val)
        return dist.sample()

    def eval_step(self, batch, batch_idx):
        states, actions, next_states, dones = HypotensionDataset.extract_s_and_a_pairs(
            batch['x_list'], state_type='all')

        probs = self.get_action_probs(states, actions)

        # Calculate per-action probability
        per_act_a = probs.new_zeros(HypotensionDataset.TOTAL_ACTIONS)
        per_act_total = probs.new_zeros(HypotensionDataset.TOTAL_ACTIONS)
        for i in range(HypotensionDataset.TOTAL_ACTIONS):
            per_act_a[i] = probs[actions == i].sum()
            per_act_total[i] = (actions == i).sum()

        return {f'per_act_a': per_act_a, f'per_act_total': per_act_total}

    def eval_epoch_end(self, outputs, prefix='val', **kwargs):
        """Log the avg of the test results"""
        per_act_a = torch.stack([x['per_act_a'] for x in outputs])
        per_act_total = torch.stack([x['per_act_total'] for x in outputs])

        per_act_acc = per_act_a.sum(dim=0) / per_act_total.sum(dim=0)
        balanced_acc = per_act_acc.mean()
        act0_acc = per_act_acc[0]
        act_others_acc = per_act_a[:, 1:].sum() / per_act_total[:, 1:].sum()

        avg_acc = per_act_a.sum() / per_act_total.sum()

        tensorboard_logs = {f'{prefix}_a': avg_acc, f'{prefix}_bal_a': balanced_acc,
                            f'{prefix}_act0_acc': act0_acc, f'{prefix}_act_others_acc': act_others_acc}
        return tensorboard_logs

    @classmethod
    def get_rs_loader(cls, args, rs=None):
        if rs is None:
            rs = RandomSearch(hparams=args, seed=args.seed)
        # rs.add_rs_hparams('sync_rate', short_name='sr', chose_from=[100, 200])
        # rs.add_rs_hparams('beta', short_name='bta', chose_from=[0.25])
        # rs.add_rs_hparams('gen_n_hidden', short_name='fnh', chose_from=[512])
        rs.add_rs_hparams('gen_n_layer', short_name='fnl', chose_from=[3, 4])
        rs.add_rs_hparams('gen_dropout', short_name='fdr', chose_from=[0.5])
        # rs.add_rs_hparams('use_bn', short_name='ubn', chose_from=[1])
        rs.add_rs_hparams('gen_lr', short_name='glr', chose_from=[4e-4, 8e-4])
        # rs.add_rs_hparams('gen_wd', short_name='gwd', chose_from=[0, 5e-6])

        # Both seems failing in matching expert
        # rs.add_rs_hparams('update_q_by_gen', short_name='uqbg', chose_from=[0., 0.5, 1.])
        # rs.add_rs_hparams('sample_stdev', short_name='sstd', chose_from=[0.])
        # rs.add_rs_hparams('ent_reg', short_name='ereg', chose_from=[1])

        # rs.add_rs_hparams('bc_kl', short_name='bc', chose_from=[2, 4, 8])
        rs.add_rs_hparams('bc_kl_anneal', short_name='bca',
                          gen=lambda hparams: -1 if hparams.bc_kl == 0
                          else rs.np_gen.choice([0.5])) # Just the first half
        return rs

    @classmethod
    def add_model_specific_args(cls, parser) -> argparse.ArgumentParser:
        """
        Adds arguments for DQN model
        Note: these params are fine tuned for Pong env
        Args:
            parent
        """
        parser.add_argument('--update_q_by_gen', default=0.5, type=float,
                            help='If > 0, then also use generated exp (simulated by GRU) to '
                                 'update Q-learning. If 0.5, only use half loss for these exps.')
        parser.add_argument('--beta', type=float, default=0.25) # Beta in soft Q-learning
        parser.add_argument('--update_gen', type=int, default=1) # Update in every iteration?
        # gru_dir = '0915_gru_l1_s97_lr0.0005_wd1e-05_bs128_nh128_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh128_anl0_ano32_adr0.0'
        gru_dir = '0915_gru_l1_s146_lr0.001_wd1e-06_bs256_nh64_nl1_dr0.0_fnh512_fnl2_fdr0.3_anh64_anl2_ano32_adr0.3'
        parser.add_argument('--ts_model_name', type=str, default=gru_dir)
        parser.add_argument("--sync_rate", type=int, default=200,
                            help="how many steps do we update the target network")
        parser.add_argument('--gen_lr', default=5e-4, type=float)
        parser.add_argument('--gen_wd', default=0., type=float)

        parser.add_argument('--gen_n_hidden', default=512, type=int)
        parser.add_argument('--gen_n_layer', default=4, type=int)
        parser.add_argument('--gen_dropout', default=0.1, type=float)

        parser.add_argument('--sample_stdev', default=0., type=float)
        parser.add_argument('--rollout', default=0, type=int,
                            help='If 1, the gen exp is roll-outted. If 0, then use one-step forward roll-out')
        parser.add_argument('--ent_reg', default=1., type=float,
                            help='In obj add the entropy regularization.')
        parser.add_argument('--bc_kl', default=5, type=float,
                            help='The coefficient for kl loss between q-learning prob and behavior cloning prob')
        parser.add_argument('--bc_kl_anneal', default=-1, type=float,
                            help='Annealing the bc kl divergence for first portion of training.'
                                 'If <= 0, no annealing!')
        parser.add_argument('--use_bn', default=1, type=int, help='Should we use batchnorm?')
        return parser


class MIMIC3BCGenerator(MIMIC3Generator):
    """ Use BC as policy generator """
    def __init__(self, hparams) -> None:
        nn.Module.__init__(self)
        self.hparams = hparams
        self.ts_model = load_best_model_from_trained_dir(self.hparams.ts_model_name)

        # Dummy optimizer
        self.dummy_params = torch.nn.Parameter(torch.tensor(1.), requires_grad=False)
        self.optimizer = torch.optim.SGD([self.dummy_params], lr=0.)
        # Cache the gen_exp for disc to speed up.
        self.gen_batch = None
        self.bc_policy = load_best_model_from_trained_dir('0909_bc_mlp_s107_lr0.002_wd1e-05_bs64_fnh384_fnl4_fdr0.5')

    def get_action_probs(self, s, a):
        '''
        s: numpy int array [B]
        a: numpy int array [B]
        '''
        log_prob = self.get_action_log_probs(s, a)
        return torch.exp(log_prob)

    def get_action_log_probs(self, s, a=None, qnet='source'):
        '''
        s: numpy int array [B]
        a: numpy int array [B]
        '''
        assert len(s.shape) == 2

        log_probs = F.log_softmax(self.bc_policy(s), dim=-1)

        if a is None:
            return log_probs
        return log_probs.gather(1, a.unsqueeze(-1)).squeeze(-1)

    def update(self, reward_disc, step, batch, epoch):
        return None

    def gen_exp(self, expert_batch, actions='agent', state_type='features', same_act_use_exp=1.):
        states, actions, new_states, dones = self._gen_exp(
            expert_batch, actions=actions, same_act_use_exp=same_act_use_exp)

        states = HypotensionDataset.extract_s_by_state_type(states, state_type)
        new_states = HypotensionDataset.extract_s_by_state_type(new_states, state_type)
        return states, actions, new_states, dones

    def _gen_exp(self, expert_batch, actions='agent', same_act_use_exp=1.):
        assert actions in ['agent', 'expert']

        x_list = expert_batch['x_list']
        if actions == 'expert' and same_act_use_exp == 1.:
            return HypotensionDataset.extract_s_and_a_pairs(x_list, 'all')

        x_len = [v.size(0) for v in x_list]
        x_pad = pad_sequence(x_list, batch_first=True)
        generated_exp = self.ts_model.simulate(
            x_pad,
            actions_fn=self.sample_action if actions == 'agent' else None,
            sample_stdev=0.,
            rollout=0)
        generated_exp = [ge[:the_len] for ge, the_len in zip(generated_exp, x_len)]

        states, actions, new_states, dones = HypotensionDataset.extract_s_and_a_pairs(generated_exp, 'all')
        # Only do one-step ahead prediction
        exp_states, exp_actions, exp_new_states, exp_dones = \
            HypotensionDataset.extract_s_and_a_pairs(x_list, 'all')

        if same_act_use_exp > 0.:
            is_same = (actions == exp_actions).all(dim=-1)
            new_states[is_same] = (1. - same_act_use_exp) * new_states[is_same] \
                                  + same_act_use_exp * exp_new_states[is_same]

        if actions == 'agent': # Cache
            self.gen_batch = exp_states, actions, new_states, dones
        return exp_states, actions, new_states, dones

    def sample_action(self, s):
        '''
        Sample actions from the DQN.

        states: pytorch array of [B, D]
        '''
        logits = self.bc_policy(s)
        dist = Categorical(logits=logits)
        return dist.sample()

    def eval_step(self, batch, batch_idx):
        states, actions, next_states, dones = HypotensionDataset.extract_s_and_a_pairs(
            batch['x_list'], state_type='all')

        probs = self.get_action_probs(states, actions)

        # Calculate per-action probability
        per_act_a = probs.new_zeros(HypotensionDataset.TOTAL_ACTIONS)
        per_act_total = probs.new_zeros(HypotensionDataset.TOTAL_ACTIONS)
        for i in range(HypotensionDataset.TOTAL_ACTIONS):
            per_act_a[i] = probs[actions == i].sum()
            per_act_total[i] = (actions == i).sum()

        return {f'per_act_a': per_act_a, f'per_act_total': per_act_total}

    def eval_epoch_end(self, outputs, prefix='val', **kwargs):
        """Log the avg of the test results"""
        per_act_a = torch.stack([x['per_act_a'] for x in outputs])
        per_act_total = torch.stack([x['per_act_total'] for x in outputs])

        per_act_acc = per_act_a.sum(dim=0) / per_act_total.sum(dim=0)
        balanced_acc = per_act_acc.mean()
        act0_acc = per_act_acc[0]
        act_others_acc = per_act_a[:, 1:].sum() / per_act_total[:, 1:].sum()

        avg_acc = per_act_a.sum() / per_act_total.sum()

        tensorboard_logs = {f'{prefix}_a': avg_acc, f'{prefix}_bal_a': balanced_acc,
                            f'{prefix}_act0_acc': act0_acc, f'{prefix}_act_others_acc': act_others_acc}
        return tensorboard_logs

    @classmethod
    def get_rs_loader(cls, args, rs=None):
        if rs is None:
            rs = RandomSearch(hparams=args, seed=args.seed)
        return rs

    @classmethod
    def add_model_specific_args(cls, parser) -> argparse.ArgumentParser:
        """
        Adds arguments for DQN model
        Note: these params are fine tuned for Pong env
        Args:
            parent
        """
        # gru_dir = '0915_gru_l1_s97_lr0.0005_wd1e-05_bs128_nh128_nl1_dr0.0_fnh384_fnl2_fdr0.15_anh128_anl0_ano32_adr0.0'
        gru_dir = '0915_gru_l1_s146_lr0.001_wd1e-06_bs256_nh64_nl1_dr0.0_fnh512_fnl2_fdr0.3_anh64_anl2_ano32_adr0.3'
        parser.add_argument('--ts_model_name', type=str, default=gru_dir)
        return parser
