from params.top_view_trainer.sbpd.uniform_grid.top_view_trainer_params import create_params as create_top_view_trainer_params


def create_params():
    p = create_top_view_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 3  # (x, y ,theta)

    # Change the learning rate and number of samples
    p.trainer.lr = 1e-4
    p.trainer.num_samples = int(20e3)

    # Change the checkpoint path
    p.trainer.ckpt_path = '/home/vtolani/Documents/Projects/visual_mpc/logs/sbpd/topview/nn_waypoint/uniform_grid/train_full_episode_20k_image_data_source/hard_goals/session_2018-12-12_10-51-48/checkpoints/ckpt-13'

    p.test.number_tests = 1
    return p
