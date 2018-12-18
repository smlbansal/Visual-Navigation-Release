from params.rgb_trainer.sbpd.projected_grid.rgb_trainer_params import create_params as create_rgb_trainer_params


def create_params():
    p = create_rgb_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 3  # (x, y ,theta)

    # Change the learning rate and num_samples
    p.trainer.lr = 1e-4
    p.trainer.num_samples = int(50e3)

    # Change the checkpoint
    p.trainer.ckpt_path = '/home/vtolani/Documents/Projects/visual_mpc/logs/sbpd/rgb/nn_waypoint/projected_grid/predict_image_space/normalize_waypt_coord/train_full_episode_50k/session_2018-12-16_19-39-53/checkpoints/ckpt-20'

    return p
