import numpy as np
from dotmap import DotMap
import matplotlib.pyplot as plt
import cv2
import os


params = DotMap()
params.nn = DotMap(z_near=0.01, z_far=20.0, h=0.8, t=45.0*np.pi/180.0, half_fov_hor=45.0*np.pi/180.0, half_fov_vert=45.0*np.pi/180.0)
params.robot = DotMap(z_near=0.01, z_far=20.0, h=0.8, t=23.0*np.pi/180.0, half_fov_hor=35.0*np.pi/180.0, half_fov_vert=17.5*np.pi/180.0)


def compute_four_image_points():
    # Compute the four corners in the world space that correspond to the new camera parameters
    # w1 ------------ w2
    #    ------------
    # w4 ------------ w3
    #

    w1 = np.array([params.robot.z_near * np.tan(params.robot.half_fov_hor),
                   params.robot.z_near * np.tan(params.robot.half_fov_vert), params.robot.z_near])
    w2 = np.array([-params.robot.z_near * np.tan(params.robot.half_fov_hor),
                   params.robot.z_near * np.tan(params.robot.half_fov_vert), params.robot.z_near])
    w3 = np.array([-params.robot.z_near * np.tan(params.robot.half_fov_hor),
                   -params.robot.z_near * np.tan(params.robot.half_fov_vert), params.robot.z_near])
    w4 = np.array([params.robot.z_near * np.tan(params.robot.half_fov_hor),
                   -params.robot.z_near * np.tan(params.robot.half_fov_vert), params.robot.z_near])
    
    # Now let's rotate these points by the change in the tilt amount (we will assume that the new camera parameters
    # are such that the field of view of the modifed camera is always within the fov of the original camera). Also, tilt
    # of the new camera is assumed to be less that the original tilt
    diff_tilt = params.nn.t - params.robot.t
    R = np.array([[1., 0., 0.],
                  [0., np.cos(diff_tilt), -np.sin(diff_tilt)],
                  [0., np.sin(diff_tilt), np.cos(diff_tilt)]])
    
    w1 = R.dot(w1)
    w2 = R.dot(w2)
    w3 = R.dot(w3)
    w4 = R.dot(w4)
    
    # Compute the four corners in the pixel space corresponding to above world space coordinates
    # (image is assumed to be going from (0, 0) to (1, 1))
    # p4 ------------ p3
    #     ----------
    #    p1 ------ p2
    
    p1 = project_to_image_plane_as_per_nn_parameters(w1)
    p2 = project_to_image_plane_as_per_nn_parameters(w2)
    p3 = project_to_image_plane_as_per_nn_parameters(w3)
    p4 = project_to_image_plane_as_per_nn_parameters(w4)

    # Setting that works with original (x, y) order
    return np.stack((p3, p4, p1, p2), axis=0).astype("float32")
    
    # # Setting that works with flipped (x, y) order
    # return np.stack((p3, p2, p1, p4), axis=0).astype("float32")


def project_to_image_plane_as_per_nn_parameters(world_pt, flip_image=True, normalize=True):
    # Flip_x flip the x_coordinates to make an image behind the camera looks like
    # the images plane is in the front of the camera
    wx_image_n = -params.nn.z_near * world_pt[0] / max(world_pt[2], params.nn.z_near)
    wy_image_n = -params.nn.z_near * world_pt[1] / max(world_pt[2], params.nn.z_near)

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
    
    folder_name = '/home/somilb/Documents/Projects/visual_mpc/tmp/changing_camera_params/' \
                 'width_240_height_240_fovh_90.0_fovv_90.0_tilt_-45'
    
    image_files = ['0.png', '1.png', '2.png', '3.png', '4.png']
    
    # Read the image
    images = []
    for file in image_files:
        images.append(cv2.imread(os.path.join(folder_name, file)))
    images = np.stack(images, axis=0)
        
    # Input image
    maxWidth_input = 224
    maxHeight_input = 224
    
    # Output image
    maxWidth_output = 224
    maxHeight_output = 224
    
    image_points = compute_four_image_points()
    rect = maxHeight_input * image_points
    dst = np.array([
        [0, 0],
        [maxWidth_output - 1, 0],
        [maxWidth_output - 1, maxHeight_output - 1],
        [0, maxHeight_output - 1]], dtype="float32")

    M = cv2.getPerspectiveTransform(rect, dst)
    
    # plt.figure(1)
    # plt.subplot(111)
    # plt.imshow(image)
    # for i in range(4):
    #     plt.plot(rect[i, 0], rect[i, 1], 'bo')
    #     plt.annotate('p%i' %(i+1), (rect[i, 0], rect[i, 1]))
    # plt.savefig("original_with_points.pdf")
    
    # cv2.imwrite("original.png", image)
    counter = 0
    method_name = 'method5'
    new_dir = os.path.join(folder_name, 'warped_images_' + method_name)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for image in image_files:
        warped = cv2.warpPerspective(images[counter], M, (maxWidth_output, maxHeight_output))
        filename = os.path.join(new_dir, 'warped_' + method_name + '_' + image)
        cv2.imwrite(filename, warped)
        counter += 1


if __name__ == '__main__':
    perform_perspective_projection()
