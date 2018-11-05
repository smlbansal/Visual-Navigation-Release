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
    # p = parse_params(params.load_params())
    return p


def parse_params(p):
    return p

#Todo: Probably delete this
def parse_params_old(p):
    """Parse a parameter file to compute other relevant parameters
    for experiments."""
    import pdb; pdb.set_trace()
    # Map Origin and size
    origin_x = int(p.map_bounds[0][0]/p.dx)
    origin_y = int(p.map_bounds[0][1]/p.dx)
    p.map_origin_2 = np.array([origin_x, origin_y], dtype=np.int32)

    Nx = int((p.map_bounds[1][0] - p.map_bounds[0][0])/p.dx)
    Ny = int((p.map_bounds[1][1] - p.map_bounds[0][1])/p.dx)
    p.map_size_2 = [Nx, Ny]

    # Horizons in timesteps
    p.episode_horizon = int(np.ceil(p.episode_horizon_s/p.dt))
    p.ks = [int(np.ceil(x/p.dt)) for x in p.planning_horizons_s]
    p.control_horizon = int(np.ceil(p.control_horizon_s/p.dt))

    C = tf.diag(p.lqr_quad_coeffs, name='lqr_coeffs_quad')
    c = tf.constant(p.lqr_linear_coeffs, name='lqr_coeffs_linear',
                    dtype=tf.float32)

    p.cost_params = {'C_gg': C, 'c_g': c}

    dx = p.planner_params.dx
    num_theta_bins = ensure_odd(p.planner_params.num_theta_bins)
    # Check implied batch size for uniform sampling
    if p.planner_params.mode == 'uniform':
        x0, y0 = p.waypoint_bounds[0]
        xf, yf = p.waypoint_bounds[1]
        # Make sure these are odd so the origin is included (for turning waypoints)
        nx = ensure_odd(int((xf-x0)/dx))
        ny = ensure_odd(int((yf-y0)/dx))
        p.planner_params.waypt_x_params = [x0, xf, nx]
        p.planner_params.waypt_y_params = [y0, yf, ny]
        p.planner_params.waypt_theta_params = [-np.pi/2, np.pi/2, num_theta_bins]
        # subtract one since the [0, 0, 0] point will be ignored
        p.n = int(nx*ny*num_theta_bins-1)

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
