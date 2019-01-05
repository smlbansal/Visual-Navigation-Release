import numpy as np

IMAGENET_MEAN_1113 = np.array([[[[0.485, 0.456, 0.406]]]], dtype=np.float32)
IMAGENET_STD_1113 = np.array([[[[0.229, 0.224, 0.225]]]], dtype=np.float32)

IMAGENET_MEAN_INT_1113 = np.array([[[[103.939, 116.779, 123.68]]]], dtype=np.float32)


def rgb_normalize(raw_data):
    """
    Normalize rgb images to [0, 1] range
    """
    raw_data['img_nmkd'] = raw_data['img_nmkd']/255.
    return raw_data


def resnet50_normalize(raw_data):
    """
    Normalize pixel values to [-1, 1] range
    """

    raw_data['img_nmkd'] = raw_data['img_nmkd']/127.5
    raw_data['img_nmkd'] -= 1.

    return raw_data

def resnet50_normalize_v1(raw_data):
    """
    Normalize image by imagenet statistics
    """
    
    raw_data['img_nmkd'] = raw_data['img_nmkd']/255.
    raw_data['img_nmkd'] = (raw_data['img_nmkd']-IMAGENET_MEAN_1113) / IMAGENET_STD_1113 

    return raw_data

def resnet50_normalize_v2(raw_data):
    """
    Caffe normalization. Switch RGB to BGR
    and normalize by imagenet statistics
    """
  
    # rgb to bgr
    raw_data['img_nmkd'] = raw_data['img_nmkd'][:, :, :, ::-1]
    raw_data['img_nmkd'] = (raw_data['img_nmkd']-IMAGENET_MEAN_INT_1113) 

    return raw_data
