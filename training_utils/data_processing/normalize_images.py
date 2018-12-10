def rgb_normalize(raw_data):
    """
    Normalize rgb images to [0, 1] range
    """
    img_nmkd = raw_data['img_nmkd']
    n, m, k, d = img_nmkd.shape
    raw_data['img_nmkd'] = img_nmkd/255.
    return raw_data

