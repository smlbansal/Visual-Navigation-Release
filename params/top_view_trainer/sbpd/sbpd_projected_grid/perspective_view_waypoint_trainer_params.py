from params.top_view_trainer.sbpd.sbpd_projected_grid.perspective_view_trainer_params import create_params as create_perspective_view_trainer_params


def create_params():
    p = create_perspective_view_trainer_params()

    # The number of model outputs
    p.model.num_outputs = 3  # (x, y, theta) waypoint

    # The learning rate
    p.trainer.lr = 1e-4
    p.trainer.num_samples = int(20e3)
    
    # Checkpoint
    p.trainer.ckpt_path = '/home/vtolani/Documents/Projects/visual_mpc/logs/sbpd/topview/nn_waypoint/projected_grid/predict_3d/train_full_episode_20k/session_2018-12-17_23-14-29/checkpoints/ckpt-20'
    #'/home/vtolani/Documents/Projects/visual_mpc/logs/sbpd/topview/nn_waypoint/projected_grid/predict_3d/train_full_episode_20k/session_2018-12-17_16-03-21/checkpoints/ckpt-7'

    return p
