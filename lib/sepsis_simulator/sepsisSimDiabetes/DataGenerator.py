import numpy as np
import scipy.stats as stat
from scipy.special import logsumexp

from .Action import Action
from .MDP import MDP_DICT

NUM_OBS = 5 #total dim of obs space: hr, sbp, O2, gluc, diabetes


def update_belief(belief,obs,action,log_T,O,obs_mask=None): 
    """
    update & return new belief from old belief, new obs & reward, and action
    
    NOTE: In settings where true model not known, 
    these T and O should be estimated and *not* the truth.
    """
    O_means = O[0]; O_sds = O[1] #O_dims,n_S,n_A       

    if obs_mask is None:
        log_obs = np.sum(stat.norm.logpdf(
            obs,O_means[:,:,action].T,O_sds[:,:,action].T),1) #S'
    else:
        log_obs = np.sum(obs_mask*stat.norm.logpdf(
            obs,O_means[:,:,action].T,O_sds[:,:,action].T),1) #S'

    #T: S' x S 
    lb = np.log(belief+1e-16) # S
    log_T_b = log_T[:,:,action] + lb[None,:]# S' x S

    log_b = log_obs + logsumexp(log_T_b,1)
    return np.exp(log_b - logsumexp(log_b))


'''
Simulates data generation from an MDP
'''
class DataGenerator(object):
    def __init__(self, seed=1789, mdp='linear'):
        self.seed = seed
        self.mdp = mdp
        self.rng = np.random.RandomState(seed)

        self.mdp_cls = MDP_DICT[mdp]

    def simulate_PBVI_policy(self, num_iters, max_num_steps,
            policy, POMDP_params, PBVI_temp = None, 
            policy_idx_type='full', p_diabetes=0.2,
            output_state_idx_type='full', obs_sigmas=0.3, 
            meas_probs=1.0):
        '''
        policy corresponds to result of running PBVI on learned POMDP:
            policy[0] is set of alpha vectors, policy[1] is action probs per alpha vec

            this is only used to evaluate our learned policies, NOT generate data to run with
        '''

        policy_alpha_vecs = policy[0]
        policy_act_probs = policy[1]
        pi,T,O,R = POMDP_params
        log_T = np.log(T+1e-16)

        # Set the default value of states / actions to negative -1,
        # corresponding to None

        iter_rewards = np.zeros((num_iters, max_num_steps))
        iter_lengths = np.zeros((num_iters), dtype=int)

        mdp = self.mdp_cls(init_state_idx=None, # Random initial state
                           policy_array=None, policy_idx_type=policy_idx_type,
                           p_diabetes=p_diabetes, seed=self.seed)

        for itr in range(num_iters):

            # MDP will generate the diabetes index as well
            mdp.state = mdp.get_new_state()
            this_diabetic_idx = mdp.state.diabetic_idx

            belief = np.copy(pi) #initial belief is prior over states...

            # add some obs noise to true states, then cache...
            this_state_vec = mdp.state.get_phi_vector()
            this_obs = this_state_vec + obs_sigmas*self.rng.normal(0,1,NUM_OBS)

            obs_mask = self.rng.uniform(0,1,NUM_OBS) <= meas_probs

            #update belief before taking first action...
            init_action = 0
            belief = update_belief(belief,this_obs,init_action,log_T,O,obs_mask)

            for step in range(max_num_steps):

                #select action by following PBVI-learned policy
                if PBVI_temp is None: #deterministic policy
                    b_alpha = np.dot(policy_alpha_vecs,belief)
                    alpha = np.argmax(b_alpha)
                    action = np.argmax(policy_act_probs[alpha])
                else: #stochastic policy defined by softmax over alphas w/ temperature 
                    b_alpha = np.dot(policy_alpha_vecs,belief)/PBVI_temp    
                    exp_alpha = np.exp(b_alpha-np.max(b_alpha))
                    alpha_probs = exp_alpha/np.sum(exp_alpha)
                    
                    action_probs = np.sum(alpha_probs[:,None] * policy_act_probs,0)
                    action = np.where(self.rng.multinomial(1,action_probs,1))[1][0]
            
                # Take the action, new state is property of the MDP
                step_reward = mdp.transition(Action(action_idx = action))

                # add some obs noise to true states, then cache...
                this_state_vec = mdp.state.get_phi_vector()
                this_obs = this_state_vec + obs_sigmas*self.rng.normal(0,1,NUM_OBS)

                #randomly mask out part of observations according to fixed probs
                obs_mask = self.rng.uniform(0,1,NUM_OBS) <= meas_probs
                belief = update_belief(belief,this_obs,action,log_T,O,obs_mask)

                if step_reward != 0:
                    iter_rewards[itr, step] = step_reward
                    iter_lengths[itr] = step+1
                    break

            if step == max_num_steps-1:
                iter_lengths[itr] = max_num_steps

        return iter_rewards

    def simulate(self, policy, num_iters, max_num_steps=20,
                 policy_idx_type='full', p_diabetes=0.2,
                 output_state_idx_type='full', obs_sigmas=0.3,
                 meas_probs=1.0):
        '''
        params:
            num_iters: how many trajectores to sample
            max_num_steps: max time point T
            policy: use this policy to execute action. It's an numpy array
                with size [n_states, n_actions].
            policy_idx_type: should use full. It means ignore the hidden
                state diabetes in the diabetes MDP.
            obs_sigmas: the Gaussian noise added to the observation
            meas_probs: the probability of keeping the observation. This
                will results in random masks in the return
        return:
            states: np array of [num_iters, (T+1)]
                the integer array specifying which state it is.
            actions: np array of [num_iters, T]
                the integer array specifying which action it is. (0~7)
            lengths: np array of [num_iters]
                the integer specifying how long the trajectory is.
            rewards: np array of [num_iters, T]
                the reward value for each time
            initobs: np array of [num_iters, 5]
                the inital observation. It consists of 4 obs and 1 diabetic
            obs: np array of [num_iters, T, 5]
            initobs_mask: np array of [num_iters, 5]
                indicate which obs is masked. Not used.
            obs_mask: np array of [num_iters, T, 5]
            beh_probs: np array of [num_iters, T]
                The action probability
        '''
        assert policy is not None, "Please specify a policy"

        if type(obs_sigmas) == float:
            obs_sigmas = obs_sigmas*np.ones(NUM_OBS)

        # Set the default value of states / actions to negative -1,
        # corresponding to None
        iter_states = np.ones((num_iters, max_num_steps+1), dtype=int)*(-1)
        iter_actions = np.ones((num_iters, max_num_steps), dtype=int)*(-1)

        iter_rewards = np.ones((num_iters, max_num_steps)) *-np.inf
        iter_lengths = np.zeros((num_iters), dtype=int)

        iter_initobs = np.ones((num_iters, NUM_OBS), dtype=float)*-np.inf
        iter_obs = np.ones((num_iters, max_num_steps, NUM_OBS), dtype=float)*-np.inf

        iter_initobs_mask = np.zeros((num_iters, NUM_OBS), dtype=int)
        iter_obs_mask = np.zeros((num_iters, max_num_steps, NUM_OBS), dtype=int)

        #cache probability of each beh action
        iter_beh_probs = np.ones((num_iters, max_num_steps), dtype=float)*(-1) 

        # Record diabetes, the hidden mixture component
        iter_component = np.zeros((num_iters, max_num_steps), dtype=int)

        mdp = self.mdp_cls(init_state_idx=None, # Random initial state
                           policy_array=policy, policy_idx_type=policy_idx_type,
                           p_diabetes=p_diabetes, seed=self.seed)

        for itr in range(num_iters):
            # MDP will generate the diabetes index as well
            mdp.state = mdp.get_new_state()
            this_diabetic_idx = mdp.state.diabetic_idx
            iter_component[itr, :] = this_diabetic_idx  # Never changes

            iter_states[itr, 0] = mdp.state.get_state_idx(
                idx_type=output_state_idx_type)

            # add some obs noise to true states, then cache...
            this_state_vec = mdp.state.get_phi_vector()
            this_obs = this_state_vec + obs_sigmas*self.rng.normal(0,1,NUM_OBS)
            iter_initobs[itr, :] = this_obs

            iter_initobs_mask[itr, :] = self.rng.uniform(0,1,NUM_OBS) <= meas_probs

            for step in range(max_num_steps):
                step_action = mdp.select_actions() #policy takes action & returns Action object
                this_action_idx = step_action.get_action_idx().astype(int)

                this_from_state_idx = mdp.state.get_state_idx(
                        idx_type=output_state_idx_type).astype(int)

                # get the beh prob from the stochastic policy
                iter_beh_probs[itr,step] = policy[this_from_state_idx,this_action_idx]

                # Take the action, new state is property of the MDP
                step_reward = mdp.transition(step_action)
                this_to_state_idx = mdp.state.get_state_idx(
                        idx_type=output_state_idx_type).astype(int)

                iter_states[itr, step + 1] = this_to_state_idx
                iter_actions[itr, step] = this_action_idx
                iter_rewards[itr, step] = step_reward

                # add some obs noise to true states, then cache...
                this_state_vec = mdp.state.get_phi_vector()
                this_obs = this_state_vec + obs_sigmas*self.rng.normal(0,1,NUM_OBS)
                iter_obs[itr, step, :] = this_obs

                iter_obs_mask[itr, step, :] = self.rng.uniform(0,1,NUM_OBS) <= meas_probs

                ## No terminal state in our setup
                # if step_reward != 0:
                #     iter_rewards[itr, step] = step_reward
                #     iter_lengths[itr] = step+1
                #     break

            if step == max_num_steps-1:
                iter_lengths[itr] = max_num_steps

        return (iter_states, iter_actions, iter_lengths, iter_rewards, 
            iter_component, iter_initobs, iter_obs, iter_initobs_mask, 
            iter_obs_mask, iter_beh_probs)
