import os, torch
import time
import pandas as pd
from os.path import exists as pexists
import numpy as np


def output_csv(data_dict, the_path, order=None, delimiter=',', float_precision=5,
               default_val=-999):
    if the_path.endswith('.tsv'):
        delimiter = '\t'

    is_file_exists = os.path.exists(the_path)
    keys = list(data_dict.keys())
    if order is not None:
        keys = order + [k for k in keys if k not in order]

    col_title = delimiter.join([str(k) for k in keys])
    if not is_file_exists:
        with open(the_path, 'a+') as op:
            print(col_title, file=op)
    else:
        old_col_title = open(the_path, 'r').readline().strip()
        if col_title != old_col_title:
            old_order = old_col_title.split(delimiter)
            no_key = [k for k in old_order if k not in keys]
            if len(no_key) > 0:
                print('The data_dict does not have the '
                      'following old keys: %s' % str(no_key))

            additional_keys = [k for k in keys if k not in old_order]
            if len(additional_keys) > 0:
                print('The data_dict has following additional '
                      'keys %s. Create new cols and fill with %s'
                      % (str(additional_keys), str(default_val)))

            df_exist = pd.read_csv(the_path, delimiter=delimiter)
            for k in additional_keys:
                df_exist[k] = default_val

            # New order is keys (the new passed-in order) + no_key
            keys += no_key
            df_exist = df_exist[keys]

            num = 0
            while pexists(the_path + f'.old{num}'):
                num += 1
            os.rename(the_path, the_path + f'.old{num}')
            df_exist.to_csv(the_path, sep=delimiter, index=False)

    vals = []
    for k in keys:
        val = data_dict.get(k, default_val)
        if isinstance(val, torch.Tensor) and val.ndim == 0:
            val = val.item()
        if isinstance(val, float):
            val = ("{:.%dg}" % float_precision).format(val)
        vals.append(str(val))

    with open(the_path, 'a+') as op:
        print(delimiter.join(vals), file=op)


class Timer:
    def __init__(self, name, remove_start_msg=True):
        self.name = name
        self.remove_start_msg = remove_start_msg

    def __enter__(self):
        self.start_time = time.time()
        print('Run "%s".........' % self.name, end='\r' if self.remove_start_msg else '\n')

    def __exit__(self, exc_type, exc_val, exc_tb):
        time_diff = float(time.time() - self.start_time)
        time_str = '{:.1f}s'.format(time_diff) if time_diff >= 1 else '{:.0f}ms'.format(time_diff * 1000)

        print('Finish "{}" in {}'.format(self.name, time_str))


def make_dir_permanent(name):
    '''
    In Vaughan server, the model is periodically cleaned. To avoid it,
    we can store it in the /hdd partition instead of using soft link
    '''

    from shutil import copy
    from os.path import exists as pexists, join as pjoin

    assert os.path.islink(pjoin('logs', name)), 'Not a link: %s' % (pjoin('logs', name))

    os.makedirs('logs/temp_dir', exist_ok=True)
    for f in os.listdir(pjoin('logs', name)):
        copy(pjoin('logs', name, f), 'logs/temp_dir/', follow_symlinks=False)
        os.remove(pjoin('logs', name, f))
    os.unlink(pjoin('logs', name))

    os.rename('logs/temp_dir', pjoin('logs', name))


def average_GAM_dfs(all_dfs):
    first_df = all_dfs[0]
    if len(all_dfs) == 1:
        return first_df

    all_feat_idx = first_df.feat_idx.values.tolist()
    for i in range(1, len(all_dfs)):
        all_feat_idx += all_dfs[i].feat_idx.values.tolist()
    all_feat_idx = set(all_feat_idx)

    results = []
    for feat_idx in all_feat_idx:
        if isinstance(feat_idx, np.int64):
            feat_idx = int(feat_idx)
        # all_dfs_with_this_feat_idx = []

        all_dfs_with_this_feat_idx = [
            df[df.feat_idx.apply(lambda x: np.all(x == feat_idx))].iloc[0] for df in all_dfs
            if df.feat_idx.apply(lambda x: np.all(x == feat_idx)).any()
        ]

        all_ys = [df.y for df in all_dfs_with_this_feat_idx]
        if len(all_ys) == 0:
            import pdb; pdb.set_trace()

        if len(all_ys) < len(all_dfs):  # Not every df has the index
            diff = len(all_dfs) - len(all_ys)
            # print(f'Add {diff} times 0 arr in {feat_idx}')
            for _ in range(diff):
                all_ys.append(np.zeros(len(all_ys[0])).tolist())

        y_mean = np.mean(all_ys, axis=0)
        y_std = np.std(all_ys, axis=0)

        row = all_dfs_with_this_feat_idx[0]
        result = {
            'feat_name': row.feat_name,
            'feat_idx': row.feat_idx,
            'x': row.x,
            'y': y_mean,
            'y_std': y_std,
        }
        if 'counts' in row:
            result['counts'] = row.counts
            result['importance'] = np.average(np.abs(y_mean), weights=row.counts)

        results.append(result)

    df = pd.DataFrame(results)
    # sort it
    df['tmp'] = df.feat_idx.apply(
        lambda x: x[0] * 1e10 + x[1] * 1e5 if isinstance(x, tuple) else int(x))
    df = df.sort_values('tmp').drop('tmp', axis=1)
    df = df.reset_index(drop=True)
    return df



