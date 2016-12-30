"""
Some persistence methods when dealing with long computation times.
"""

import pickle
import logging
from sklearn.base import RegressorMixin
from xgboost import XGBRegressor,XGBClassifier

import config
from model.models import Simulator


def _get_label(object_):

    if isinstance(object_, dict):
        return 'dataset'
    if isinstance(object_, RegressorMixin) or \
            isinstance(object_, XGBRegressor) or isinstance(object_, XGBClassifier):
        return 'model'
    if isinstance(object_, Simulator):
        return 'simulator'
    else:
        logging.error(type(object_))
        raise NotImplementedError


def pickle_dump(object_, simulator):
    label = _get_label(object_)
    pickle.dump(object_, open(config.data_store_path +
                              'predictions/{0}_{1}_with_{2}_shifts.p'.format(label, simulator.target,
                                                                             simulator.shift), "wb"))


def pickle_load(label, simulator):
    return pickle.load(open(config.data_store_path +
                            'predictions/{0}_{1}_with_{2}_shifts.p'.format(label,
                                                                           simulator.target,
                                                                           simulator.shift), "rb"))
