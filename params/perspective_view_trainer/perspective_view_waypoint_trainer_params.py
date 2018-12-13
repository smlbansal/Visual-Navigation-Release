from params.perspective_view_trainer.perspective_view_trainer_params import create_params as create_perspective_view_trainer_params


def create_params():
    p = create_perspective_view_trainer_params()

    # The number of model outputs
    p.model.num_outputs = 3  # (x, y, theta) waypoint

    # The learning rate
    p.trainer.lr = 1e-4
    
    # Checkpoint
    p.trainer.ckpt_path = '/home/vtolani/Documents/Projects/visual_mpc/logs/circular_obstacle_map/nn_waypoint/perspective_grid/train_full_episode_20k/session_2018-12-07_10-13-45/checkpoints/ckpt-40'

    # Test params
    p.test.simulate_expert = True
    return p