from params.rgb_trainer.rgb_trainer_params import create_params as create_rgb_trainer_params
from training_utils.data_processing.rgb_preprocess import preprocess as preprocess_image_data


def create_params():
    p = create_rgb_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 3  # (x, y ,theta)

    # Change the learning rate
    p.trainer.lr = 1e-4
    p.trainer.num_samples = int(20e3)

    # Change the data_dir
    p.data_creation.data_dir = '/home/ext_drive/somilb/data/training_data/sbpd/topview_full_episode_random_v1_100k'
    p.data_creation.img_data_dir = '/home/ext_drive/somilb/data/training_data/sbpd/topview_full_episode_random_v1_100k/rgb_image_data_2018-12-10_11-48-20'

    # Change the Data Processing
    p.data_processing.input_processing_function = preprocess_image_data 
    return p
