from imgaug import augmenters as iaa
import numpy as np
import cv2


def custom_augmenter_v1(sometimes):
    seq = iaa.Sequential(
        [
            # execute 0 to 5 of the following (less important) augmenters per image
            # don't execute all of them, as that would often be way too strong
            iaa.SomeOf((0, 4),
                [
                    sometimes(iaa.Superpixels(p_replace=(0, 1.0), n_segments=(100, 200))), # convert images into their superpixel representation
                    iaa.OneOf([
                        iaa.GaussianBlur((0, 3.0)), # blur images with a sigma between 0 and 3.0
                        iaa.AverageBlur(k=(2, 7)), # blur image using local means with kernel sizes between 2 and 7
                        # iaa.MedianBlur(k=(3, 11)), # blur image using local medians with kernel sizes between 2 and 7
                        iaa.MotionBlur(k=(3, 7)) # blur image using motion blur
                        # with angle between [-90, 90] and kernel size between 2 and 7
                    ]),
                    iaa.Sharpen(alpha=(0, 1.0), lightness=(0.75, 1.5)), # sharpen images
                    iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)), # emboss images
                    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.25), # add gaussian noise to images
                    iaa.OneOf([
                        iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
                    ]),
                    iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)
                    iaa.AddToHueAndSaturation((-20, 20)), # change hue and saturation
                    # either change the brightness of the whole image (sometimes
                    # per channel) or change the brightness of subareas
                    iaa.OneOf([
                        iaa.Multiply((0.5, 1.5), per_channel=0.5),
                        iaa.FrequencyNoiseAlpha(
                            exponent=(-4, 0),
                            first=iaa.Multiply((0.5, 1.5), per_channel=True),
                            second=iaa.ContrastNormalization((0.5, 2.0))
                        )
                    ]),
                    iaa.ContrastNormalization((0.5, 2.0), per_channel=0.5), # improve or worsen the contrast
                    iaa.Grayscale(alpha=(0.0, 1.0)),
                    sometimes(iaa.ElasticTransformation(alpha=(0.5, 2.5), sigma=0.25)), # move pixels locally around (with random strengths)
                ],
                random_order=True
            )
        ],
        random_order=True
    )
    return seq


def custom_augmenter_v2():
    # This is a replica of the distortion function in the old Visual-MPC code
    seq = iaa.Sequential(
        [
            # Change brightness of images (by -20 to 20 of original value)
            iaa.Add((-20, 20), per_channel=0.5),
            
            # Change the Hue and Saturation of the image
            iaa.AddToHueAndSaturation((-20, 20), per_channel=0.5),
    
            # Improve or worsen the contrast
            iaa.ContrastNormalization((0.5, 1.5), per_channel=0.5),
            
        ], random_order=True)
    return seq


def custom_augmenter_v3(params):
    # Create a sequencer to apply the typical distortions
    sometimes = lambda aug: iaa.Sometimes(params.p, aug)
    seq1 = custom_augmenter_v1(sometimes)
    
    # Create a distortion for fov
    # TODO (Somil): The base renderer parameters are hard-coded for now. This should be changed to accept the SBPD
    # renderer parameters directly.
    base_tilt = np.float32(45.0*np.pi/180.0)
    base_fov = np.float32(45.0*np.pi/180.0)
    base_f = np.float32(0.01)
    tilt_min = np.float32(20.0*np.pi/180.0)
    tilt_max = np.float32(50.0*np.pi/180.0)
    # The FOVs are all half field of views
    fov_hor_min = np.float32(25.0*np.pi/180.0)
    fov_hor_max = np.float32(35.0*np.pi/180.0)
    fov_ver_min = np.float32(20.0*np.pi/180.0)
    fov_ver_max = np.float32(30.0*np.pi/180.0)
    
    def create_four_image_points(tilts_n, fovs_hor_n, fovs_ver_n):
        n = tilts_n.shape[0]
        
        tan_fovs_hor_n = np.tan(fovs_hor_n)
        tan_fovs_ver_n = np.tan(fovs_ver_n)
    
        # Create four world points in the space of the new fov
        wX_n4 = np.stack((-tan_fovs_hor_n, tan_fovs_hor_n, tan_fovs_hor_n, -tan_fovs_hor_n), axis=1)
        wY_n4 = np.stack((-tan_fovs_ver_n, -tan_fovs_ver_n, tan_fovs_ver_n, tan_fovs_ver_n), axis=1)
        wZ_n4 = np.ones((n, 4), dtype=np.float32)
        wXYZ_n431 = np.stack((wX_n4, wY_n4, wZ_n4), axis=2)[:, :, :, None] * base_f

        # Define the rotation matrix
        diff_tilt_n = base_tilt - tilts_n
        float32_1 = np.float32(1.)
        float32_0x5 = np.float32(0.5)
        R_n133 = np.zeros((n, 1, 3, 3), dtype=np.float32)
        R_n133[:, 0, 0, 0] = float32_1
        R_n133[:, 0, 1, 1] = np.cos(diff_tilt_n)
        R_n133[:, 0, 1, 2] = -np.sin(diff_tilt_n)
        R_n133[:, 0, 2, 1] = np.sin(diff_tilt_n)
        R_n133[:, 0, 2, 2] = np.cos(diff_tilt_n)

        # Project the points back in the old fov
        wXYZ_n431 = np.matmul(R_n133, wXYZ_n431)
        
        # Project the points to the image space
        wx_image_n4 = wXYZ_n431[:, :, 0, 0] / (np.maximum(wXYZ_n431[:, :, 2, 0], base_f) * np.tan(base_fov))
        wy_image_n4 = wXYZ_n431[:, :, 1, 0] / (np.maximum(wXYZ_n431[:, :, 2, 0], base_f) * np.tan(base_fov))

        return np.stack((float32_0x5 * (wx_image_n4 + float32_1), float32_0x5 * (wy_image_n4 + float32_1)), axis=2)
    
    def fov_and_tilt_distortion(images_nmkd):
        # Figure out the image shape
        n, m, k, d = images_nmkd.shape
        
        # Create a random list of fov and tilts
        tilts = np.random.uniform(tilt_min, tilt_max, n).astype(np.float32)
        fovs_hor = np.random.uniform(fov_hor_min, fov_hor_max, n).astype(np.float32)
        fovs_ver = np.random.uniform(fov_ver_min, fov_ver_max, n).astype(np.float32)
        
        # Create a list of four image points to distort
        image_points_n42 = create_four_image_points(tilts, fovs_hor, fovs_ver)
        dst = np.array([[0, 0], [k - 1, 0], [k - 1, m - 1], [0, m - 1]], dtype="float32")
        
        warped_images = []
        for i in range(n):
            M = cv2.getPerspectiveTransform(m * image_points_n42[i], dst)
            warped_images.append(cv2.warpPerspective(images_nmkd[i], M, (k, m)))
        
        return np.stack(warped_images, axis=0)
        
    return [seq1, fov_and_tilt_distortion]


def basic_image_distortor(params):
    """
    Basic distortion function to distort a series of images.
    :param params: a set of parameters for initializing the distortion function
    :return: distorted images
    """
    # Create an augmentation object
    if params.version == 'v1':
        # Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
        # e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
        sometimes = lambda aug: iaa.Sometimes(params.p, aug)
        seq = [custom_augmenter_v1(sometimes)]
    elif params.version == 'v2':
        seq = [custom_augmenter_v2()]
    elif params.version == 'v3':
        seq = custom_augmenter_v3(params)
    else:
        raise NotImplementedError
    return seq
