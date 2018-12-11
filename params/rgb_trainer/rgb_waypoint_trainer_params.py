from params.rgb_trainer.rgb_trainer_params import create_params as create_rgb_trainer_params


def create_params():
    p = create_rgb_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 3  # (x, y ,theta)

    # Change the learning rate
    p.trainer.lr = 1e-4
    p.trainer.num_samples = int(50e3)

    # Change the checkpoint
    p.trainer.ckpt_path = '/home/vtolani/Documents/Projects/visual_mpc/logs/sbpd/rgb/nn_waypoint/train_full_episode_20k/session_2018-12-10_23-13-59/checkpoints/ckpt-20'
    return p
