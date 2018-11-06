from dotmap import DotMap

def load_params():
    """Common parameters for all experiments."""
    p = DotMap()
    p.seed = 1 #for tensorflow and numpy
    return p

def parse_params(p):
    return p