from params.turtlebot.turtlebot_navigator_params import create_params as create_turtlebot_params


def create_params():
    p = create_turtlebot_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 3  # (x, y ,theta)

    # Change the checkpoint
    p.trainer.ckpt_path = '/home/vtolani/Documents/Projects/visual_mpc/logs/sbpd/rgb/nn_waypoint/uniform_grid/train_full_episode_50k/session_2018-12-12_13-34-16/checkpoints/ckpt-20'

    return p
