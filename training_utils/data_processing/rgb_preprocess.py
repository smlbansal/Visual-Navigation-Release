from training_utils.data_processing.normalize_images import rgb_normalize
from copy import deepcopy


def preprocess(raw_data):
    # make a copy of the data
    raw_data = deepcopy(raw_data)

    # normalize rgb images to [0, 1]
    raw_data = rgb_normalize(raw_data)

    # TODO: Add data augmentation here
    return raw_data
