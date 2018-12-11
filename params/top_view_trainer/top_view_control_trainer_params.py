from params.top_view_trainer.top_view_trainer_params import create_params as create_top_view_trainer_params


def create_params():
    p = create_top_view_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 60  # (v, omega) for 30 timesteps

    # Change the learning rate
    p.trainer.lr = 1e-5

    # Circular Obstacle Map
    p.data_creation.data_dir = '/home/ext_drive/somilb/data/training_data/circular_obstacle_map/topview_full_episode_100k' 

    
    # SBPD
    #p.data_creation.data_dir = '/home/ext_drive/somilb/data/training_data/sbpd/topview_full_episode_random_v1_100k'

    return p
