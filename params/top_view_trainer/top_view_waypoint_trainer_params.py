from params.top_view_trainer.top_view_trainer_params import create_params as create_top_view_trainer_params


def create_params():
    p = create_top_view_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 3  # (x, y ,theta)

    # Change the learning rate
    p.trainer.lr = 1e-4
    p.trainer.num_samples = int(20e3)

    # Change the data_dir
    
    # Circular Obstacle Map
    #p.data_creation.data_dir = '/home/ext_drive/somilb/data/training_data/circular_obstacle_map/topview_full_episode_100k' 
    #p.data_creation.img_data_dir = '/home/ext_drive/somilb/data/training_data/circular_obstacle_map/topview_full_episode_100k/occupancy_grid_image_data_2018-12-10_15-07-45'

    
    # SBPD
    p.data_creation.data_dir = '/home/ext_drive/somilb/data/training_data/sbpd/topview_full_episode_random_v1_100k'

    p.data_creation.img_data_dir = '/home/ext_drive/somilb/data/training_data/sbpd/topview_full_episode_random_v1_100k/occupancy_grid_image_data_2018-12-10_14-25-47/'


    # Change the checkpoint
    #p.trainer.ckpt_path = '/home/vtolani/Documents/Projects/visual_mpc/logs/tmp/check_circular_map/session_2018-12-10_15-22-33/checkpoints/ckpt-20'

    p.trainer.ckpt_path = '/home/vtolani/Documents/Projects/visual_mpc/logs/tmp/check_sbpd/session_2018-12-10_17-41-38/checkpoints/ckpt-20'
    return p