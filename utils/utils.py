import dotmap, commentjson
import os


def load_params(version):
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


def subplot2(plt, Y_X, sz_y_sz_x=(10,10), space_y_x=(0.1,0.1), T=False):
  Y,X = Y_X
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
