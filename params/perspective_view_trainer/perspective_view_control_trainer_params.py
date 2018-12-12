from params.perspective_view_trainer.perspective_view_trainer_params import create_params as create_perspective_view_trainer_params


def create_params():
    p = create_perspective_view_trainer_params()

    # The number of model outputs
    p.model.num_outputs = 60  # (v, omega) for 30 timesteps 

    # The learning rate
    p.trainer.lr = 1e-5

    # Checkpoint
    p.trainer.ckpt_path = '/home/vtolani/Documents/Projects/visual_mpc/logs/circular_obstacle_map/nn_control/perspective_grid/train_full_episode_20k/session_2018-12-11_11-25-11/checkpoints/ckpt-40'

    # Test params
    p.test.simulate_expert = True
    return p
