from params.top_view_trainer.circular_obstacle_map.projected_grid.perspective_view_trainer_params import create_params as create_perspective_view_trainer_params


def create_params():
    p = create_perspective_view_trainer_params()

    # The number of model outputs
    p.model.num_outputs = 3  # (x, y, theta) waypoint

    # The learning rate
    p.trainer.lr = 1e-4
    p.trainer.num_samples = int(20e3)
    
    # Checkpoint
    p.trainer.ckpt_path = '/home/vtolani/Documents/Projects/visual_mpc/logs/debug/circular_obs_map_predict_image_with_sbpd_normalized_grid_20k_samples/session_2018-12-17_14-58-53/checkpoints/ckpt-7'

    # Test params
    p.test.simulate_expert = False
    p.test.number_tests = 100
    return p
