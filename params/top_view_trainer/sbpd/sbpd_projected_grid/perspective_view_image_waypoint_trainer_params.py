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

    # Project the goal position into the image frame
    p.model.project_goal_coordinates_to_image_plane = True

    # Add an indicator telling the model whether the goal is in front
    # or behind the image plane
    p.model.include_goal_direction_indicator = True
    p.model.num_inputs.num_state_features = 3 + 2  # Goal (x, y), Goal direction ind, and vehicle
    # current speed and angular speed
    
    # Checkpoint
    p.trainer.ckpt_path = '/home/vtolani/Documents/Projects/visual_mpc/logs/sbpd/topview/nn_waypoint/projected_grid/predict_image_space/normalized_waypt_coord/goal_in_image_space/goal_direction_indicator/train_full_episode_20k/session_2018-12-20_17-44-08/checkpoints/ckpt-20'

    p.trainer.simulate_expert = False

    return p