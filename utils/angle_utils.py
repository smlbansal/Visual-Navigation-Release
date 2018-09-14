import numpy as np


# Angle normalization function
def angle_normalize(x):
    return (((x + np.pi) % (2 * np.pi)) - np.pi)