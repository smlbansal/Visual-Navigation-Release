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
