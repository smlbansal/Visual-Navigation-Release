from params.top_view_trainer.sbpd.sbpd_projected_grid.perspective_view_trainer_params import create_params as create_perspective_view_trainer_params


def create_params():
    p = create_perspective_view_trainer_params()

    # The number of model outputs
    p.model.num_outputs = 3  # (x, y, theta) waypoint

    # The learning rate
    p.trainer.lr = 1e-4
    p.trainer.num_samples = int(20e3)
    
    # Rescale NN supervision from [0, 1]
    p.model.rescale_imageframe_coordinates = True
    
    # Checkpoint
    p.trainer.ckpt_path = '/home/vtolani/Documents/Projects/visual_mpc/logs/sbpd/topview/nn_waypoint/projected_grid/predict_image_space/normalized_waypt_coord/train_full_episode_20k_newest/session_2018-12-17_16-46-40/checkpoints/ckpt-20' 

    return p
