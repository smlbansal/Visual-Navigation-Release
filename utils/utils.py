import os
import pickle
import json
import importlib
import copy
import numpy as np
import tensorflow as tf
import dotmap


def gpu_config():
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    return config


def ensure_odd(integer):
    if integer % 2 == 0:
        integer += 1
    return integer

def load_params(module_name):
    """Loads a parameter file in ./params/module_name.py"""
    module_name = 'params.{:s}'.format(module_name)
    params = importlib.import_module(module_name)
    p = params.parse_params(params.load_params())

    common_params = importlib.import_module('params.common_params')
    p.common = common_params.parse_params(common_params.load_params())
    return p

def log_dict_as_json(params, filename):
    """Save params (either a DotMap object or a python dictionary) to a file in json format"""
    with open(filename, 'w') as f:
        if isinstance(params, dotmap.DotMap):
            params = params.toDict()
        param_dict_serializable = _to_json_serializable_dict(copy.deepcopy(params))
        json.dump(param_dict_serializable, f, indent=4, sort_keys=True)


def _to_json_serializable_dict(param_dict):
    """ Converts params_dict to a json serializable dict."""
    def _to_serializable_type(elem):
        """ Converts an element to a json serializable type. """
        if isinstance(elem, np.int64) or isinstance(elem, np.int32):
            return int(elem)
        if isinstance(elem, tf.Tensor):
            return elem.numpy().tolist()
        if isinstance(elem, np.ndarray):
            return elem.tolist()
        if isinstance(elem, dict):
            return _to_json_serializable_dict(elem)
        if type(elem) is type:  # elem is a class
            return str(elem)
        else:
            return elem
    for key in param_dict.keys():
        param_dict[key] = _to_serializable_type(param_dict[key])
    return param_dict


def mkdir_if_missing(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)

#Probably can delete
def dump_to_pickle_file(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_from_pickle_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data


def subplot2(plt, Y_X, sz_y_sz_x=(10, 10), space_y_x=(0.1, 0.1), T=False):
    Y, X = Y_X
    sz_y, sz_x = sz_y_sz_x
    hspace, wspace = space_y_x
    plt.rcParams['figure.figsize'] = (X*sz_x, Y*sz_y)
    fig, axes = plt.subplots(Y, X, squeeze=False)
    plt.subplots_adjust(wspace=wspace, hspace=hspace)
    if T:
        axes_list = axes.T.ravel()[::-1].tolist()
    else:
        axes_list = axes.ravel()[::-1].tolist()
    return fig, axes, axes_list
