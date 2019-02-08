from training_utils.data_processing.normalize_images import resnet50_normalize_v2, resnet50_normalize_v1, resnet50_normalize
from copy import deepcopy


def preprocess(raw_data):
    # make a copy of the data
    raw_data = deepcopy(raw_data)

    # normalize rgb images to [-1, 1]
    raw_data = resnet50_normalize(raw_data)

    # TODO: Add data augmentation here
    return raw_data


def preprocess_v1(raw_data):
    # make a copy of the data
    raw_data = deepcopy(raw_data)

    # normalize rgb images to [0, 1] and then
    # normalize by the imagenet statistics
    raw_data = resnet50_normalize_v1(raw_data)

    # TODO: Add data augmentation here
    return raw_data


def preprocess_v2(raw_data):
    # make a copy of the data
    raw_data = deepcopy(raw_data)

    # caffe normalization: swap image channels (rgb -> bgr)
    # normalize by the imagenet statistics
    raw_data = resnet50_normalize_v2(raw_data)

    # TODO: Add data augmentation here
    return raw_data

