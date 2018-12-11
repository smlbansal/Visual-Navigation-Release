def rgb_normalize(raw_data):
    """
    Normalize rgb images to [0, 1] range
    """
    raw_data['img_nmkd'] = raw_data['img_nmkd']/255.
    return raw_data

