import numpy as np
from .sepsisSimDiabetes.DataGenerator import DataGenerator


class SepsisMCMuEstimator(object):
    """
    Estimate the mu to step through the passed in simulator
    """

    def __init__(self, seed, gamma=0.9, state_type='next', mdp='linear'):
        """
        :params phi: the phi function that encodes feature \phi(s, a).
            Usually just a concatenation.
        :params gamma: the decay gamma in RL
        :params state_type: choose from ['next', 'current']. Use current state or
            next state to do MMA. But in MMA there is just a single time-step difference...
        """
        self.seed = seed
        self.mdp = mdp
        self._gamma = gamma
        self._env = DataGenerator(seed=seed, mdp=mdp)
        self.state_type = state_type

    def fit(self, pi_eval, stochastic):
        self._pi_eval = pi_eval
        self._stochastic = stochastic

    def estimate(self, n_sample=500):
        """
        using monte carlo samples with a simulator
        """
        (_, _, _, _, _, init_observs, observs, _, _, _) = self._env.simulate(
            num_iters=n_sample,
            policy=self._pi_eval,
            policy_idx_type='full',
            obs_sigmas=0., # No noise!
        )

        states = observs
        if self.state_type == 'current':
            states = np.concatenate([init_observs[:, None, :], observs[:, :-1, :]], axis=1)

        # Our phi(s, a) is just the next state
        states[np.isinf(states)] = 0
        # ^-- [n_sample, T, 5]

        mu = 0.0
        for t in range(states.shape[1]):
            mu += states[:, t, :] * (self._gamma ** t)
        mu_mean = np.mean(mu, axis=0)
        return mu_mean
