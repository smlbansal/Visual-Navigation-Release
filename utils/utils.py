import os
import json
import copy
import numpy as np
import tensorflow as tf
import dotmap
import shutil
from dotmap import DotMap


def tf_session_config():
    config = tf.ConfigProto()

    # Allows for memory growth so the process only uses the amount of memory it needs
    config.gpu_options.allow_growth = True

    # Allows for tensors to be copied onto cpu when no cuda gpu kernel is available
    device_policy = tf.contrib.eager.DEVICE_PLACEMENT_SILENT

    tf_config = {'config': config,
                 'device_policy': device_policy}
    return tf_config
    

def ensure_odd(integer):
    if integer % 2 == 0:
        integer += 1
    return integer


def render_angle_frequency(p):
    """Returns a render angle frequency
    that looks heuristically nice on plots."""
    return int(p.episode_horizon/25)


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
            return str(elem)
    for key in param_dict.keys():
        param_dict[key] = _to_serializable_type(param_dict[key])
    return param_dict


def mkdir_if_missing(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def delete_if_exists(dirname):
    if os.path.exists(dirname):
        shutil.rmtree(dirname)


def check_dotmap_equality(d1, d2):
    """Check equality on nested dotmap objects that all keys and values match."""
    assert(len(set(d1.keys()).difference(set(d2.keys()))) == 0)
    equality = [True] * len(d1.keys())
    for i, key in enumerate(d1.keys()):
        d1_attr = getattr(d1, key)
        d2_attr = getattr(d2, key)
        if type(d1_attr) is DotMap:
            equality[i] = check_dotmap_equality(d1_attr, d2_attr) 
    return np.array(equality).all()


def configure_plotting():
    import matplotlib.pyplot as plt
    plt.style.use('ggplot')


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
