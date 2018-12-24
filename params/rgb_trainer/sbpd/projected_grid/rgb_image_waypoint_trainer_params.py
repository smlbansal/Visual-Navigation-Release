from params.rgb_trainer.sbpd.projected_grid.rgb_trainer_params import create_params as create_rgb_trainer_params


def create_params():
    p = create_rgb_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 3  # (x, y ,theta)

    # Change the learning rate and num_samples
    p.trainer.lr = 1e-4
    p.trainer.num_samples = int(50e3)

    # Rescale NN supervision from [0, 1]
    p.model.rescale_imageframe_coordinates = True
    
    # Project the goal position into the image frame
    p.model.project_goal_coordinates_to_image_plane = True

    # Add an indicator telling the model whether the goal is in front
    # or behind the image plane
    p.model.include_goal_direction_indicator = True
    p.model.num_inputs.num_state_features = 3 + 2  # Goal (x, y), Goal direction ind, and vehicle
    # current speed and angular speed

    # Change the checkpoint
    p.trainer.ckpt_path = '/home/vtolani/Documents/Projects/visual_mpc/logs/sbpd/rgb/nn_waypoint/projected_grid/predict_image_space/normalize_waypt_coord/goal_in_image_space/forward_backward_indicator/train_full_episode_50k/session_2018-12-21_11-43-19/checkpoints/ckpt-18'

    p.test.number_tests = 1
    p.test.simulate_expert = False
    return p
