from params.top_view_trainer.circular_obstacle_map.top_view_trainer_params import create_params as create_top_view_trainer_params


def create_params():
    p = create_top_view_trainer_params()

    # Change the number of model outpus
    p.model.num_outputs = 3  # (x, y ,theta)

    # Change the learning rate and number of samples
    p.trainer.lr = 1e-4
    p.trainer.num_samples = int(20e3)

    # Change the checkpoint path
    p.trainer.ckpt_path = ''
    return p
