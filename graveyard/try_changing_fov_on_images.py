import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt
import cv2


params = DotMap()
params.nn = DotMap(z_near=0.01, z_far=20.0, h=0.8, t=45.0*np.pi/180.0, half_fov_hor=45.0*np.pi/180.0, half_fov_vert=45.0*np.pi/180.0)
params.robot = DotMap(z_near=0.01, z_far=20.0, h=0.8, t=36.0*np.pi/180.0, half_fov_hor=30.0*np.pi/180.0, half_fov_vert=24.75*np.pi/180.0)


def compute_four_image_points():
    # Compute the four corners in the world space that correspond to the new camera parameters
    # w4 ------------ w3
    #     ----------
    #    w1 ------ w2
    #
    eps = 1e-8
    eff_z_near = max(params.robot.h * np.tan(0.5 * np.pi - params.robot.t - params.robot.half_fov_vert + eps),
                     params.robot.z_near)
    eff_z_far = min(params.robot.h * np.tan(0.5 * np.pi - params.robot.t + params.robot.half_fov_vert - eps),
                    params.robot.z_far)
    
    w1 = np.array([eff_z_near, -eff_z_near * np.tan(params.robot.half_fov_hor)])
    w2 = np.array([eff_z_near, eff_z_near * np.tan(params.robot.half_fov_hor)])
    w3 = np.array([eff_z_far, eff_z_far * np.tan(params.robot.half_fov_hor)])
    w4 = np.array([eff_z_far, -eff_z_far * np.tan(params.robot.half_fov_hor)])
    
    # Compute the four corners in the pixel space corresponding to above world space coordinates
    # (image is assumed to be going from (0, 0) to (1, 1))
    # p4 ------------ p3
    #     ----------
    #    p1 ------ p2
    
    p1 = project_to_image_plane_as_per_nn_parameters(w1)
    p2 = project_to_image_plane_as_per_nn_parameters(w2)
    p3 = project_to_image_plane_as_per_nn_parameters(w3)
    p4 = project_to_image_plane_as_per_nn_parameters(w4)

    # return np.stack((p4, p3, p2, p1), axis=0).astype("float32")
    return np.stack((p1, p2, p3, p4), axis=0).astype("float32")


def project_to_image_plane_as_per_nn_parameters(world_pt, flip_image=True, normalize=True):
    # Flip_x flip the x_coordinates to make an image behind the camera looks like
    # the images plane is in the front of the camera
    wx_n = world_pt[0]
    wy_n = world_pt[1]
    den = wx_n * np.cos(params.nn.t) + params.nn.h * np.sin(params.nn.t)
    wx_image_n = -params.nn.z_near * wy_n / wx_n
    wy_image_n = -params.nn.z_near * (wx_n * np.sin(params.nn.t) - params.nn.h * np.cos(params.nn.t)) / wx_n

    if flip_image:
        wx_image_n = -wx_image_n
        wy_image_n = -wy_image_n

    if normalize:
        wx_image_n = wx_image_n / (params.nn.z_near * np.tan(params.nn.half_fov_hor))
        wy_image_n = wy_image_n / (params.nn.z_near * np.tan(params.nn.half_fov_vert))

        wx_image_n = 0.5 * (wx_image_n + 1)
        wy_image_n = 0.5 * (wy_image_n + 1)

    return np.array([wx_image_n, wy_image_n])
    

def perform_perspective_projection():
    
    image_file = '/home/somilb/Documents/Projects/visual_mpc/tmp/changing_camera_params/' \
                 'width_240_height_240_fovh_90.0_fovv_90.0_tilt_-45/1.png'
    
    # Read the image
    image = cv2.imread(image_file)

    # Input image
    maxWidth_input = 224
    maxHeight_input = 224
    
    # Output image
    maxWidth_output = 320
    maxHeight_output = 240
    
    image_points = compute_four_image_points()
    rect = maxHeight_input * image_points
    dst = np.array([
        [0, 0],
        [maxWidth_output - 1, 0],
        [maxWidth_output - 1, maxHeight_output - 1],
        [0, maxHeight_output - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth_output, maxHeight_output))

    # seq = custom_augmenter_v3()
    # warped_imgaug = seq.augment_images([image.astype(np.uint8)])[0].astype(np.float32)
    # cv2.imwrite("warped_imgaug.png", warped_imgaug)
    
    # import ipdb; ipdb.set_trace()
    plt.figure(1)
    plt.subplot(111)
    plt.imshow(image)
    for i in range(4):
        plt.plot(rect[i, 0], rect[i, 1], 'bo')
    plt.savefig("original_with_points.pdf")
    
    # cv2.imwrite("original.png", image)
    cv2.imwrite("warped.png", warped)


def custom_augmenter_v3():
    from imgaug import augmenters as iaa
    # This is a replica of the distortion function in the old Visual-MPC code
    seq = iaa.Sequential([iaa.PerspectiveTransform(scale=(0.2, 0.35), keep_size=True)])
    return seq


if __name__ == '__main__':
    perform_perspective_projection()
