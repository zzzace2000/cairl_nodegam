import pandas as pd
import torch

import pickle
from os.path import exists as pexists, join as pjoin
from torch.utils.data import Dataset
import numpy as np
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence, pack_sequence
from sklearn.model_selection import train_test_split
import torch.nn.functional as F
from ..mypreprocessor import MyPreprocessor
from ..utils import Timer


class HypotensionDataset(Dataset):
    # Some facts in the preprocessing
    MAX_TIME = 24
    DISCRETIZE = 2
    NUM_VASO_BINS = 4
    NUM_FLUID_BINS = 4
    TOTAL_ACTIONS = NUM_VASO_BINS * NUM_FLUID_BINS
    FLUID_MEDIAN_IN_EACH_BIN = [0, 293, 501, 1001]
    VASO_MEDIAN_IN_EACH_BIN = [0, 3, 13.5, 37.76]

    # Here define columns to be used
    covariates = [
        'age', 'is_F', 'surg_ICU', 'is_not_white',
        'is_emergency', 'is_urgent',
    ]
    features = [
        'dbp', 'fio2', 'hr', 'map', 'sbp', 'spontaneousrr', 'spo2', 'temp',
        'urine', 'weight', 'bun', 'magnesium', 'platelets', 'sodium', 'alt', 'hct', 'po2',
        'ast', 'potassium', 'wbc',
        'bicarbonate', 'creatinine', 'lactate', 'pco2', 'bilirubin_total', 'glucose', 'inr',
        'hgb', 'GCS',
    ]
    features_ind = [f'{f}_ind' for f in features]
    # But we provide both number and discretized actions
    actions = [
        'last_vaso_1', 'last_vaso_2', 'last_vaso_3',
        'last_fluid_1', 'last_fluid_2', 'last_fluid_3',
        'total_all_prev_vasos', 'total_all_prev_fluids',
        'total_last_8hrs_vasos', 'total_last_8hrs_fluids',
    ]

    # Aggregate all cols
    all_cols = covariates + features + features_ind + actions
    col_to_idx = {n: i for i, n in enumerate(all_cols)}

    # Record the indexes of features and masks;
    # this would be used to retrieve target y and calculate loss
    len_cov, len_f = len(covariates), len(features)
    cov_idxes = list(range(len_cov))
    f_idxes = list(range(len_cov, len_cov + len_f))
    f_ind_idxes = list(range(len_cov + len_f, len_cov + 2 * len_f))
    f_actions_idxes = list(range(len_cov + 2 * len_f, len(all_cols)))
    state_idxes = cov_idxes + f_idxes + f_ind_idxes

    ##### Preprocess to normalize
    # Columns to normalize
    cols_norm = ['age', 'dbp', 'fio2', 'hr', 'map', 'sbp',
                 'spontaneousrr', 'spo2', 'temp', 'urine', 'weight', 'bun',
                 'magnesium', 'platelets', 'sodium', 'alt', 'hct', 'po2', 'ast',
                 'potassium', 'wbc', 'bicarbonate', 'creatinine', 'lactate', 'pco2',
                 'bilirubin_total', 'glucose', 'inr', 'hgb', 'GCS',
                 'total_all_prev_vasos', 'total_all_prev_fluids',
                 'total_last_8hrs_vasos', 'total_last_8hrs_fluids']

    dset = None
    pp = None

    def __init__(self, split, state_dir='./data/model-data3/', preprocess='quantile',
                 val_ratio=0.2, seed=10, debug=False, do_normalize=True, fold=0):
        '''
        Here we do 64-16-20 split for train, val and test.
        '''
        assert split in ['train', 'val', 'test', 'all']
        self.split = split
        self.state_dir = state_dir
        self.preprocess = preprocess
        self.debug = debug
        self.do_normalize = do_normalize

        # Just to save time: loading dataset into class var
        if self.__class__.dset is None:
            self.__class__.dset = self.get_dataset(
                state_dir, do_normalize=do_normalize,
            )

        fold_postfix = '' if fold == 0 else f'_{fold}'
        debug_postfix = '' if not debug else '_debug'
        if split == 'all':
            with open(pjoin(state_dir, f'test_icustay_ids{fold_postfix}{debug_postfix}.txt')) as fp:
                test_ids = [int(l.strip()) for l in fp]
            with open(pjoin(state_dir, f'train_icustay_ids{fold_postfix}{debug_postfix}.txt')) as fp:
                train_ids = [int(l.strip()) for l in fp]
            self.icustay_ids = test_ids + train_ids

        elif split == 'test':
            with open(pjoin(state_dir, f'test_icustay_ids{fold_postfix}{debug_postfix}.txt')) as fp:
                self.icustay_ids = [int(l.strip()) for l in fp]
        else:
            with open(pjoin(state_dir, f'train_icustay_ids{fold_postfix}{debug_postfix}.txt')) as fp:
                train_ids = [int(l.strip()) for l in fp]
            train_ids, val_ids = train_test_split(train_ids, test_size=val_ratio,
                                                  random_state=seed)
            self.icustay_ids = train_ids if split == 'train' else val_ids
        self.icustay_ids = np.array(self.icustay_ids)

    def get_dataset(self, state_dir, do_normalize=True):
        pp_postfix = '' if self.preprocess == 'quantile' else '_standard'
        debug_postfix = '' if not self.debug else '_debug'

        dataset_path = pjoin(state_dir, f'normalized_states{pp_postfix}{debug_postfix}.pth')
        if pexists(dataset_path) and do_normalize: # Only cache normalized states
            with Timer(f'Load cached normalized dataset: {dataset_path}'):
                return torch.load(dataset_path)

        assert pexists(pjoin(state_dir, f'all_states_extravars{debug_postfix}.p')), \
            f'No all_states_extravars{debug_postfix}.p exists in the dir {state_dir}'
        with open(pjoin(state_dir, f'all_states_extravars{debug_postfix}.p'), 'rb') as fp, Timer('Loading dataset'):
            dset = pickle.load(fp)

        with Timer('Loading preprocessor'):
            if self.__class__.pp is None:
                self.__class__.pp = self.get_preprocessor(dset, state_dir)

        with Timer('Normalized data'):
            new_dset = {}
            for i, data in dset.items():
                data = data.copy()
                data = data.loc[:, self.all_cols]
                # Do log-transform and normalize in these columns
                if do_normalize:
                    data = self.__class__.pp.transform(data)
                data = torch.from_numpy(data.values.astype(np.float32))
                new_dset[i] = data
            del dset

        if not pexists(dataset_path) and do_normalize:
            torch.save(new_dset, dataset_path)
        return new_dset

    def get_preprocessor(self, dset, state_dir, refresh=False):
        pp_postfix = '' if self.preprocess == 'quantile' else '_standard'

        path = pjoin(state_dir, f'pp{pp_postfix}.pkl')
        if pexists(path) and (not refresh):
            try:
                with open(path, 'rb') as fp:
                    pp = pickle.load(fp)
                if pp.num_features != len(self.all_cols):  # Our definition is changed
                    pp = self.get_preprocessor(dset, state_dir, refresh=True)
            except EOFError:
                pp = self.get_preprocessor(dset, state_dir, refresh=True)
            return pp

        assert not self.debug, 'In debug mode we do not generate preprocessor'
        pp = self.gen_preprocessor(dset)
        with open(path, 'wb') as op:
            pickle.dump(pp, op)
        return pp

    def gen_preprocessor(self, dset):
        with Timer('Generate new preprocessor...'):
            all_states = []
            for ID in dset:
                this_states = dset[ID]
                all_states.append(this_states)
            all_states = pd.concat(all_states)

            all_states = all_states[self.all_cols]

            pp = MyPreprocessor(cols_norm=self.cols_norm, way=self.preprocess)
            pp.fit(all_states)
        return pp

    def __len__(self):
        return len(self.icustay_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        return self.dset[self.icustay_ids[idx]].clone()

        # data = self.__class__.dset[self.icustay_ids[idx]].copy()
        # data = data.loc[:, self.all_cols]
        # # Do log-transform and normalize in these columns
        # data = self.pp.transform(data)
        # return torch.from_numpy(data.values.astype(np.float32)).clone()

    def update_act_summary(self, states, actions):
        '''
        Outputs next-state action summary given cur states and actions

        s: [B, D]. The state at time t
        actions: [B, 8]: has all the actions history taken. Useful to update the
            last_8hours_vaso
        return:
            Generate t+1 action states0909_bc_s55_lr0.001_wd1e-05_bs128_nh64_nl1_dr0.0_fnh256_fnl4_fdr0.5
        '''
        assert actions.shape[-1] == self.NUM_VASO_BINS + self.NUM_FLUID_BINS
        assert states.shape[-1] == len(self.all_cols)

        vaso_disc_ids, fluid_disc_ids = 0, 0
        for i in range(1, self.NUM_VASO_BINS):
            vaso_disc_ids += i * actions[:, i]
            fluid_disc_ids += i * actions[:, self.NUM_VASO_BINS + i]

        def replace_val(arr, cond1, cond2, val):
            tmp = arr[cond1]
            tmp[:, cond2] = val
            arr[cond1] = tmp

        def add_val(arr, cond1, cond2, val):
            tmp = arr[cond1]
            tmp2 = tmp[:, cond2] + val
            tmp2[tmp2 < 0] = 0. # No neg value
            tmp[:, cond2] = tmp2
            arr[cond1] = tmp

        def process(last_states, disc_ids, name='fluid'):
            ## First, handle the last_fluid field by the action taken
            # Handle the zero index: just zero out all features
            replace_val(
                last_states,
                cond1=(disc_ids == 0),
                cond2=[self.col_to_idx[f'last_{name}_{i}'] for i in range(1, 4)],
                val=0.)

            median_vals = getattr(self, f'{name.upper()}_MEDIAN_IN_EACH_BIN')
            for i in range(1, 4):
                if not (disc_ids == i).any():
                    continue
                replace_val(last_states, cond1=(disc_ids == i),
                            cond2=self.col_to_idx[f'last_{name}_{i}'],
                            val=1.)
                # last_states[disc_ids == i, self.col_to_idx[f'last_{name}_{i}']] = 1
                # Zero out other action indexes
                other_idxes = [self.col_to_idx[f'last_{name}_{j}'] for j in range(1, 4) if j != i]
                replace_val(last_states, cond1=(disc_ids == i),
                            cond2=other_idxes,
                            val=0)

            ## Then handle the action summary total_all_prev_vasos and total_last_8hrs_vasos
            # Convert to original space
            orig_last_states = self.pp.inverse_transform(last_states)

            total_idx = self.col_to_idx[f'total_all_prev_{name}s']
            last8_idx = self.col_to_idx[f'total_last_8hrs_{name}s']
            for i in range(1, 4):
                add_val(orig_last_states, cond1=(disc_ids == i),
                        cond2=total_idx,
                        val=median_vals[i])
                add_val(orig_last_states, cond1=(disc_ids == i),
                        cond2=last8_idx,
                        val=median_vals[i])

            # Modify features: 'total_last_8hrs_vasos', 'total_last_8hrs_fluids'
            if states.shape[1] >= (8 // self.DISCRETIZE):  # 8 hours before
                prev_8hour_state = states[:, -(8 // self.DISCRETIZE), :]
                for i in range(1, 4):
                    treat_ids = prev_8hour_state[:, self.col_to_idx[f'last_{name}_{i}']]
                    if not (treat_ids == 1).any():
                        continue

                    # Add action value to total vasos
                    add_val(orig_last_states, cond1=(treat_ids == 1),
                            cond2=last8_idx,
                            val=-median_vals[i])

            last_states = self.pp.transform(orig_last_states)
            return last_states

        last_states = states[:, -1, :].clone()
        last_states = process(last_states, vaso_disc_ids, 'vaso')
        last_states = process(last_states, fluid_disc_ids, 'fluid')

        return last_states

    @classmethod
    def extract_s_and_a_pairs(cls, x_list, state_type='all'):
        '''
        data: a pytorch of size [B, T, D] where D is the feature
        state_type: chosen from ['all', 'states', 'features'].
            'all' means all the states + past action summary.
            'states' means removing action related attributes and
                keeps both cov and features
            'features' means the time-varying part of the features

        return
            - states: the observed states up to T-1
            - actions: the actions
        '''
        x_len = [v.size(0) for v in x_list]
        x_pad = pad_sequence(x_list, batch_first=True)

        states = cls.extract_cur_s(x_pad, state_type=state_type)
        actions = cls.extract_cur_a(x_pad, form='act_idx')
        next_states = cls.extract_next_s(x_pad, state_type=state_type)

        # Construct the
        is_valid = states.new_zeros(*states.shape[:2]).bool()
        for idx, l in enumerate(x_len):
            is_valid[idx, :(l-1)] = True

        states = states[is_valid]
        actions = actions[is_valid]
        next_states = next_states[is_valid]

        dones = states.new_zeros(states.shape[0], dtype=torch.bool)
        dones[torch.cumsum(torch.tensor(x_len).to(dones.device) - 1, dim=0) - 1] = 1
        return states, actions, next_states, dones

    @classmethod
    def extract_cur_s(cls, x_pad, state_type='features'):
        tmp = cls.extract_s_by_state_type(x_pad, state_type)
        return tmp[:, :-1]

    @classmethod
    def extract_next_s(cls, x_pad, state_type='features'):
        tmp = cls.extract_s_by_state_type(x_pad, state_type)
        return tmp[:, 1:]

    @classmethod
    def extract_s_by_state_type(cls, x, state_type='features'):
        assert state_type in ['features', 'states', 'all'], f'Unknown state_type: {state_type}'

        if isinstance(x, pd.DataFrame):
            if state_type == 'features':
                return x.iloc[:, cls.f_idxes]
            if state_type == 'states':
                return x.iloc[:, cls.state_idxes]
            if state_type == 'all':
                return x

        if state_type == 'features':
            return x[..., cls.f_idxes]
        if state_type == 'states':
            return x[..., cls.state_idxes]
        if state_type == 'all':
            return x
        raise NotImplementedError(f'No such state_type {state_type}')

    @classmethod
    def get_state_dim_by_type(cls, state_type):
        if state_type == 'states':
            input_dim = len(HypotensionDataset.state_idxes)
        elif state_type == 'features':
            input_dim = len(HypotensionDataset.f_idxes)
        elif state_type == 'all':
            input_dim = len(HypotensionDataset.all_cols)
        else:
            raise NotImplementedError(f'No such state_type {state_type}')
        return input_dim

    @classmethod
    def get_state_names_by_type(cls, state_type):
        if state_type == 'states':
            names = cls.covariates + cls.features + cls.features_ind
        elif state_type == 'features':
            names = cls.features
        elif state_type == 'all':
            names = HypotensionDataset.all_cols
        else:
            raise NotImplementedError(f'No such state_type {state_type}')
        return names

    @classmethod
    def extract_cur_a(cls, x_pad, form='twohot'):
        assert form in ['act_idx', 'twohot', 'onehot']

        col_idxes = [cls.col_to_idx[k]
                     for k in ['last_vaso_1', 'last_vaso_2', 'last_vaso_3',
                               'last_fluid_1', 'last_fluid_2', 'last_fluid_3']]
        cur_act = x_pad[:, 1:, col_idxes]
        if form == 'twohot':
            vaso_0, fluid_0 = (1. - cur_act[..., :3].sum(dim=-1, keepdim=True)), \
                              (1. - cur_act[..., 3:].sum(dim=-1, keepdim=True))
            cur_act = torch.cat([vaso_0, cur_act[..., :3], fluid_0, cur_act[..., 3:]], dim=-1)
            return cur_act

        # Encode vaso and fluid as action index (0~15).
        act_idx = 0
        for i in range(1, 4):
            act_idx += cur_act[..., (i-1)] * i * cls.NUM_FLUID_BINS
            act_idx += cur_act[..., (i+2)] * i
        act_idx = act_idx.long()

        if form == 'act_idx':
            return act_idx

        # 'onehot'
        return F.one_hot(act_idx, num_classes=cls.NUM_VASO_BINS * cls.NUM_FLUID_BINS)

    @classmethod
    def convert_act_idx_to_twohot(cls, act_idx):
        '''
        Convert the action index (e.g. 14) to a two-hot format
        (e.g. [0, 0, 0, 1, 0, 0, 1, 0])
        '''

        vaso_idx = act_idx // cls.NUM_FLUID_BINS
        fluid_idx = act_idx % cls.NUM_FLUID_BINS

        result = torch.zeros(act_idx.shape[0], cls.NUM_VASO_BINS + cls.NUM_FLUID_BINS, device=act_idx.device)
        for i in range(cls.NUM_VASO_BINS):
            result[vaso_idx == i, i] = 1.
        for i in range(cls.NUM_FLUID_BINS):
            result[fluid_idx == i, i + cls.NUM_VASO_BINS] = 1.
        return result

    @classmethod
    def collate_fn(cls, xx):
        return {'x_list': xx}

    @classmethod
    def make_loader(cls, split, batch_size, debug=False, data_kwargs={}, **kwargs):
        if debug:
            kwargs['num_workers'] = 0

        dataset = cls(split=split, debug=debug, **data_kwargs)

        dataloader = DataLoader(dataset=dataset,
                                batch_size=batch_size,
                                collate_fn=cls.collate_fn,
                                **kwargs)
        return dataloader



class HypotensionWithBCProbDataset(HypotensionDataset):
    # 16-dimension
    marginal_prob = [
        0.6563, 0.0229, 0.0271, 0.0225, 0.1012, 0.0029, 0.0084, 0.0113, 0.0684,
        0.0022, 0.0045, 0.0069, 0.0516, 0.0029, 0.0039, 0.0072]

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bc_probs = torch.load(pjoin(self.state_dir, 'bc_probs.pkl'))

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.item()

        x_list = super().__getitem__(idx)
        bc_prob = self.bc_probs[self.icustay_ids[idx]].clone()
        return {'x_list': x_list, 'bc_prob': bc_prob}

    @classmethod
    def get_marginal_prob(cls, device='cuda'):
        if not torch.is_tensor(cls.marginal_prob):
            cls.marginal_prob = torch.tensor(cls.marginal_prob)
        cls.marginal_prob = cls.marginal_prob.to(device)
        return cls.marginal_prob

    @classmethod
    def collate_fn(cls, xx):
        x_list = [x['x_list'] for x in xx]
        bc_prob = [x['bc_prob'] for x in xx]
        return {'x_list': x_list, 'bc_prob': bc_prob}
