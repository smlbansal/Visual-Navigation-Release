from params.rgb_trainer.sbpd.projected_grid.rgb_trainer_params import create_params as create_rgb_trainer_params


def create_params():
    p = create_rgb_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 60  # (v, omega) for 30 timesteps

    # Change the learning rate and num_samples
    p.trainer.lr = 1e-5
    p.trainer.num_samples = int(50e3)

    # Change the checkpoint
    p.trainer.ckpt_path = '/home/vtolani/Documents/Projects/visual_mpc/logs/sbpd/rgb/nn_control/sbpd_projected_grid/train_full_episode_50k/session_2018-12-19_11-46-43/checkpoints/ckpt-20'

    return p
