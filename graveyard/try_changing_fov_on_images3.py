import numpy as np
from dotmap import DotMap
import time
import cv2
import os
    

def perform_perspective_projection():
    
    folder_name = '/home/somilb/Documents/Projects/visual_mpc/tmp/changing_camera_params/' \
                 'width_240_height_240_fovh_90.0_fovv_90.0_tilt_-45'
    
    image_files = ['0.png', '1.png', '2.png', '3.png', '4.png']
    
    # Read the image
    images = []
    for file in image_files:
        images.append(cv2.imread(os.path.join(folder_name, file)))
    images = np.stack(images, axis=0)
    
    # Create a batch of images for test
    images_batch = [images]
    for i in range(12):
        images_batch.append(images)
    images_batch = np.concatenate(images_batch, axis=0)
    
    from training_utils.data_processing.distort_images import basic_image_distortor
    params = DotMap(p=0.1, version='v3')
    image_distortor = basic_image_distortor(params)

    t1 = time.time()
    warped_images = image_distortor[1](images_batch)
    t2 = time.time()
    print(t2-t1)
    
    counter = 0
    method_name = 'new_code'
    new_dir = os.path.join(folder_name, 'warped_images_' + method_name)
    if not os.path.exists(new_dir):
        os.makedirs(new_dir)
    for image in image_files:
        filename = os.path.join(new_dir, 'warped_' + method_name + '_' + image)
        cv2.imwrite(filename, warped_images[counter])
        counter += 1


if __name__ == '__main__':
    perform_perspective_projection()
