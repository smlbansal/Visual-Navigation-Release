from params.top_view_trainer.top_view_trainer_params import create_params as create_top_view_trainer_params


def create_params():
    p = create_top_view_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 3  # (x, y ,theta)

    # Change the learning rate
    p.trainer.lr = 1e-4
    p.trainer.num_samples = int(20e3)

    # Change the data_dir
    p.data_creation.data_dir = '/home/ext_drive/somilb/data/training_data/sbpd/topview_full_episode_random_v1_100k'
    p.data_creation.img_data_dir = '/home/ext_drive/somilb/data/training_data/sbpd/topview_full_episode_random_v1_100k/tmp_rgb_image_data_2018-12-07_16-13-57'
    return p
