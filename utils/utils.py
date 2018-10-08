import dotmap
import pickle
import os
import importlib
import json
import numpy as np
import tensorflow as tf


def load_params(version):
    module_name = 'params.params_{}'.format(version)
    params = importlib.import_module(module_name)
    p = params.load_params()
    return p


def load_params_json(version):
    """Load parameters from
    ./params/params_version.json
    into a dotmap object
    """
    import commentjson
    base_dir = './params'
    filename = '{}/params_{}.json'.format(base_dir, version)
    assert(os.path.exists(filename))
    with open(filename) as f:
        params = commentjson.load(f)
        p = dotmap.DotMap(params)
    return p

def log_params(params, filename):
    with open(filename, 'w') as f:
        param_dict_serializable = _to_json_serializable_dict(params.toDict())
        json.dump(param_dict_serializable, f, indent=4, sort_keys=True)

def _to_json_serializable_dict(param_dict):
    """ Converts params_dict to a json serializable dict."""
    def _to_serializable_type(elem):
        """ Converts an element to a json serializable type. """
        if isinstance(elem, tf.Tensor):
            return elem.numpy().tolist() 
        if isinstance(elem, np.ndarray):
            return elem.tolist()
        if isinstance(elem, dict):
            return _to_json_serializable_dict(elem)
        if type(elem) is type: #elem is a class
            return str(elem)
        else:
            return elem
    for key in param_dict.keys():
        param_dict[key] = _to_serializable_type(param_dict[key])
    return param_dict

def mkdir_if_missing(dirname):
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def dump_to_pickle_file(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def load_from_pickle_file(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f)
    return data

# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


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
