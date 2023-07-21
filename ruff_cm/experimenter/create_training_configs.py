# -*- coding: utf-8 -*-
"""
Created on Mon Oct  3 15:25:58 2022

@author: hdx
The main codes are adapted from robert yang
"""

import numpy as np
import os
from ruamel import yaml
from copy import deepcopy

from ..utils import hash_string


def load_config(path):
    with open(path, "r", encoding='utf-8') as file:
        param = yaml.round_trip_load(file, preserve_quotes=True)
    return param


def save_config(param, folder, filename):
    if not os.path.exists(folder):
        os.makedirs(folder)
    with open(folder + filename + ".yml", 'w', encoding="utf-8") as file:
        yaml.dump(param, file, Dumper=yaml.RoundTripDumper)


def create_config(base_config, config_ranges, saved_folder, mode, name_keys=None):
    base_config = load_config(base_config)
    configs, configs_name = vary_config(base_config, config_ranges, mode, name_keys)
    for i in range(len(configs)):
        save_config(configs[i], saved_folder, configs_name[i])


def vary_config(base_config, config_ranges, mode, name_keys=None):
    """
    Args:
        base_config: dict, a base configuration
        config_ranges: a dictionary of experimental parameters to vary,
            keys are parameter names, values are lists of parameter values to test
            config_ranges = {
                'param1': [val1, val2, ...],
                'param2': [val3, val4, ...],
            }
        name_keys: a list of keys to include in the name
        mode: str, can take 'combinatorial', 'sequential', and 'control'
            'combinatorial': loops over Cartesian product of parameters
            'sequential': loops over parameters together sequentially
            'control': loops over the value of parameters once a time independently
    Return:
        configs: a list of config dict [config1, config2, ...]
        configs_name: a list of config name [config1_name, config2_name, ...]
    """
    if mode == 'combinatorial':
        _vary_config = _vary_config_combinatorial
    elif mode == 'sequential':
        _vary_config = _vary_config_sequential
    elif mode == 'control':
        _vary_config = _vary_config_control
    else:
        raise ValueError('Unknown mode {}'.format(str(mode)))
    configs, config_diffs = _vary_config(base_config, config_ranges)
    configs_name = autoname(configs, config_diffs, name_keys)
    return configs, configs_name


def _vary_config_combinatorial(base_config, config_ranges):
    # Unravel the input index
    keys = config_ranges.keys()
    dims = [len(config_ranges[k]) for k in keys]
    n_max = int(np.prod(dims))

    configs, config_diffs = list(), list()
    for i in range(n_max):
        config_diff = dict()
        indices = np.unravel_index(i, shape=dims)
        # Set up new config
        for key, index in zip(keys, indices):
            config_diff[key] = config_ranges[key][index]
        config_diffs.append(config_diff)

        new_config = deepcopy(base_config)
        new_config.update(config_diff)
        configs.append(new_config)
    return configs, config_diffs


def _vary_config_sequential(base_config, config_ranges):
    keys = config_ranges.keys()
    dims = [len(config_ranges[k]) for k in keys]
    n_max = dims[0]

    configs, config_diffs = list(), list()
    for i in range(n_max):
        config_diff = dict()
        for key in keys:
            config_diff[key] = config_ranges[key][i]
        config_diffs.append(config_diff)

        new_config = deepcopy(base_config)
        new_config.update(config_diff)
        configs.append(new_config)

    return configs, config_diffs


def _vary_config_control(base_config, config_ranges):
    # Unravel the input index
    keys = list(config_ranges.keys())
    dims = [len(config_ranges[k]) for k in keys]
    n_max = int(np.sum(dims))

    configs, config_diffs = list(), list()
    for i in range(n_max):
        index = i
        for j, dim in enumerate(dims):
            if index >= dim:
                index -= dim
            else:
                break

        config_diff = dict()
        key = keys[j]
        config_diff[key] = config_ranges[key][index]
        config_diffs.append(config_diff)

        new_config = deepcopy(base_config)
        new_config.update(config_diff)
        configs.append(new_config)
    return configs, config_diffs


def autoname(configs, config_diffs, name_keys=None):
    """Helper function for automatically naming models based on configs.
    Args:
        configs: a list of config dict [config1, config2, ...]
        config_diffs: a list of config diff dict [config_diff1, config_diff2, ...]
        name_keys: a list of keys to include in the name
    """
    configs_name = list()
    for config, config_diff in zip(configs, config_diffs):
        name = ''
        for key, val in config_diff.items():
            if isinstance(val, list) or isinstance(val, tuple):
                str_val = ''
                for cur in val:
                    str_val += str(cur)
            else:
                str_val = str(val)
            # if name_keys is not None, then only include keys in name_keys
            if (name_keys is None) or (key in name_keys):
                str_key = str(key).replace("_", "")  # delete '_'
                if str_key == 'OPTIMPARAMS':  # deal with special case
                    name = name + 'lr' + '-' + str(config['OPTIM_PARAMS']['lr']) + '_'
                    name = name + 'WD' + '-' + str(config['OPTIM_PARAMS']['weight_decay']) + '_'
                elif str_key == 'PATH':  # deal with special case
                    name = name + hash_string(config['PATH'], uid="", truncate=8) + '_'
                else:
                    name = name + str_key + '-' + str_val + '_'
        configs_name.append(name[:-1])  # get rid of the last '_'
    return configs_name
