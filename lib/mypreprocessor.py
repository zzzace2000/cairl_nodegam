from sklearn.preprocessing import QuantileTransformer, StandardScaler
import pandas as pd
import numpy as np
import torch


class MyPreprocessor(object):
    '''
    It implements a quantile transformation on subset of featrues.

    It supports inverse transfrom on pytorch tensor, unlike sklearn coltransformer
    '''

    def __init__(self, cols_norm, way='quantile', random_state=1377,
                 n_quantiles=1000, quantile_noise=1e-3):
        assert way in ['quantile', 'standard']

        self.way = way
        self.random_state = random_state
        self.cols_norm = cols_norm
        self.n_quantiles = n_quantiles
        self.quantile_noise = quantile_noise

        self.feature_names = None
        self.num_features = None
        self.cols_norm_idxes = None
        self.transformer = None
        self.X_shape = None
        self.torch_device = None
        self.in_cols = None

    def fit(self, X):
        assert isinstance(X, pd.DataFrame), 'X is not a dataframe! %s' % type(X)
        self.feature_names = X.columns
        self.num_features = len(self.feature_names)
        self.cols_norm_idxes = [idx for idx, f in enumerate(self.feature_names) if f in self.cols_norm]

        quantile_train = X.copy()
        quantile_train = quantile_train[self.cols_norm]

        if self.quantile_noise:
            r = np.random.RandomState(self.random_state)
            stds = np.std(quantile_train.values, axis=0, keepdims=True)
            noise_std = self.quantile_noise / np.maximum(stds, self.quantile_noise)
            quantile_train += noise_std * r.randn(*quantile_train.shape)

        if self.way == 'quantile':
            self.transformer = QuantileTransformer(random_state=self.random_state,
                                          n_quantiles=self.n_quantiles,
                                          output_distribution='normal',
                                          copy=False)
        else:
            self.transformer = StandardScaler(copy=False)

        self.transformer.fit(quantile_train)

    def subset_transform(self, X, columns):
        '''
        Given a dataframe X that's a subset of original X, only transform those columns
        '''
        X, datatype = self.to_numpy(X)

        # First create a large numpy of original size
        X_temp = np.zeros((X.shape[0], len(self.cols_norm)))

        common_cols = [c for c in columns if c in self.cols_norm]
        idxes_map = {c: idx for idx, c in enumerate(self.cols_norm)}
        idxes_map2 = {c: idx for idx, c in enumerate(columns)}

        X_temp[:, [idxes_map[c] for c in common_cols]] = X[:, [idxes_map2[c] for c in common_cols]]
        X_temp = self.transformer.transform(X_temp)

        X_ret = X.copy()
        X_ret[:, [idxes_map2[c] for c in common_cols]] = X_temp[:, [idxes_map[c] for c in common_cols]]

        X_ret = self.to_type(X_ret, datatype)
        return X_ret

    def transform(self, X):
        assert self.transformer is not None, 'Not call fit() before'

        X, datatype = self.to_numpy(X)
        assert X.shape[-1] == self.num_features, f'Passed in {X.shape[-1]} != {self.num_features}'

        X_subset = X[:, self.cols_norm_idxes]
        X_subset = self.transformer.transform(X_subset)

        X = X.copy()

        X[:, self.cols_norm_idxes] = X_subset

        X = self.to_type(X, datatype)
        return X

    def inverse_transform(self, X):
        assert self.transformer is not None, 'Not call fit() before'

        X, datatype = self.to_numpy(X)
        assert X.shape[-1] == self.num_features, f'Passed in {X.shape[-1]} != {self.num_features}'

        X_subset = X[:, self.cols_norm_idxes]
        X_subset = self.transformer.inverse_transform(X_subset)

        X = X.copy()
        X[:, self.cols_norm_idxes] = X_subset

        X = self.to_type(X, datatype)
        return X

    def to_numpy(self, X):
        datatype = 'numpy'
        if isinstance(X, pd.DataFrame):
            self.in_cols = X.columns
            X = X.values
            datatype = 'pandas'
        elif isinstance(X, torch.Tensor):
            datatype = 'torch'
            self.torch_device = X.device
            with torch.no_grad():
                X = X.cpu().numpy()

        if len(X.shape) > 2: # 3d tensor in time-series
            self.X_shape = X.shape
            X = X.reshape((-1, self.num_features))

        return X, datatype

    def to_type(self, X, datatype):
        if datatype == 'pandas':
            return pd.DataFrame(X, columns=self.in_cols)
        if datatype == 'torch':
            X = torch.from_numpy(X)
            X = X.to(self.torch_device)
            self.torch_device = None

            if self.X_shape is not None:
                X = X.reshape(*self.X_shape)
                self.X_shape = None
            return X

        # numpy
        if self.X_shape is not None:
            X = X.reshape(self.X_shape)
            self.X_shape = None
        return X
