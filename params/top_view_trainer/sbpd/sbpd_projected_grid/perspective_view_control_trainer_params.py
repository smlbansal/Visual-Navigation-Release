from params.top_view_trainer.sbpd.sbpd_projected_grid.perspective_view_trainer_params import create_params as create_top_view_trainer_params 


def create_params():
    p = create_top_view_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 60  # (v, omega) for 30 timesteps

    # Change the learning rate
    p.trainer.lr = 1e-5
    p.trainer.num_samples = int(20e3)

    # Change the checkpoint path
    p.trainer.ckpt_path = '/home/vtolani/Documents/Projects/visual_mpc/logs/sbpd/topview/nn_control/projected_grid/train_full_episode_20k/session_2018-12-19_08-45-53/checkpoints/ckpt-20'

    p.test.simulate_expert = False
    return p
