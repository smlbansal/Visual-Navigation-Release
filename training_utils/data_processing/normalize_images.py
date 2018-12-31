import numpy as np

IMAGENET_MEAN_1113 = np.array([[[[0.485, 0.456, 0.406]]]], dtype=np.float32)
IMAGENET_STD_1113 = np.array([[[[0.229, 0.224, 0.225]]]], dtype=np.float32)


def rgb_normalize(raw_data):
    """
    Normalize rgb images to [0, 1] range
    """
    raw_data['img_nmkd'] = raw_data['img_nmkd']/255.
    return raw_data


def resnet50_normalize(raw_data):
    """
    Swap image channels RGB -> BGR
    Normalize channels by the imagenet mean
    """

    raw_data['img_nmkd'] = raw_data['img_nmkd']/127.5
    raw_data['img_nmkd'] -= 1.

    # Converr RGB to BGR
    #raw_data['img_nmkd'] = raw_data['img_nmkd'][..., ::-1]

    # Normalize by the imagenet mean
    #imagenet_mean_bgr_1113 = [103.939, 116.779, 123.68]

    # Normalize rgb images to [0, 1]
    #raw_data['img_nmkd'] = raw_data['img_nmkd']/255.

    # Standardize the images by imagenet mean and std dev
    #raw_data['img_nmkd'] = (raw_data['img_nmkd'] - IMAGENET_MEAN_1113) / IMAGENET_STD_1113
    return raw_data

