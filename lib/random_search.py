from collections import namedtuple
from pytorch_lightning.utilities import AttributeDict
import argparse
import numpy as np
import copy


class RandomSearch(object):
    def __init__(self, hparams, seed=123):
        if isinstance(hparams, argparse.Namespace):
            hparams = vars(hparams)

        self.hparams = AttributeDict(**hparams)
        self.all_rs = []
        self.seed = seed
        if seed is not None:
            self.np_gen = np.random.RandomState(seed)
        else:
            self.np_gen = np.random

    def add_rs_hparams(self, name, chose_from=None, gen=None, short_name=None):
        assert chose_from is not None or gen is not None
        if short_name is None:
            short_name = name
        if chose_from is not None:
            gen = lambda x: self.np_gen.choice(chose_from)

        self.all_rs.append(AttributeDict(
            name=name, gen=gen, short_name=short_name
        ))

    def __iter__(self):
        while True:
            cur_hparams = copy.copy(self.hparams)
            cur_name = []
            for rs in self.all_rs:
                if rs.gen is not None:
                    v = rs.gen(cur_hparams)
                    if isinstance(v, np.integer):
                        v = int(v)
                    cur_hparams[rs.name] = v

                the_val = str(cur_hparams[rs.name])
                cur_name.append(rs.short_name + the_val)
            cur_name = '_'.join(cur_name)

            # Reorder hparams by putting rs earlier...
            ret_hparams = AttributeDict()
            for rs in self.all_rs:
                ret_hparams[rs.name] = cur_hparams[rs.name]
            ret_hparams.update(cur_hparams)

            ret_hparams.name = f'{ret_hparams.name}_{cur_name}'
            yield ret_hparams

    def ignore(self, user_hparams):
        self.all_rs = [r for r in self.all_rs if r.name not in user_hparams]
