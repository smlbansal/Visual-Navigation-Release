import dotmap, commentjson
import pickle
import os

def load_params():
    from params.params_v1 import load_params
    p = load_params()
    return p


def load_params_json(version):
    """Load parameters from
    ./params/params_version.json
    into a dotmap object
    """
    base_dir = './params'
    filename = '%s/params_%s.json'%(base_dir, version)
    assert(os.path.exists(filename))
    with open(filename) as f:
        params = commentjson.load(f)
        p = dotmap.DotMap(params)
    return p


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
