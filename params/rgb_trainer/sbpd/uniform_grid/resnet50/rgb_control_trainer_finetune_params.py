from params.rgb_trainer.sbpd.uniform_grid.resnet50.rgb_trainer_finetune_params import create_params as create_rgb_trainer_params


def create_params():
    p = create_rgb_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 60  # (v, omega) for 30 timesteps

    # Change the learning rate and num_samples
    p.trainer.lr = 1e-5
    p.trainer.num_samples = int(50e3)

    # Change the checkpoint
    p.trainer.ckpt_path = ''

    return p
