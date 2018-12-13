from params.top_view_trainer.circular_obstacle_map.top_view_trainer_params import create_params as create_top_view_trainer_params


def create_params():
    p = create_top_view_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 60  # (v, omega) for 30 timesteps

    # Change the learning rate
    p.trainer.lr = 1e-5
    p.trainer.num_samples = int(20e3)

    # Change the checkpoint path
    p.trainer.ckpt_path = ''
    return p
