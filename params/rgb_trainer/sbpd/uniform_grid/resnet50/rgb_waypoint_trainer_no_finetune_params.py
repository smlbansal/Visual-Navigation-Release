from params.rgb_trainer.sbpd.uniform_grid.resnet50.rgb_trainer_no_finetune_params import create_params as create_rgb_trainer_params


def create_params():
    p = create_rgb_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 3  # (x, y ,theta)

    # Change the learning rate and num_samples
    p.trainer.lr = 1e-4
    p.trainer.num_samples = int(50e3)

    p.trainer.callback_frequency = 20

    # Change the checkpoint
    p.trainer.ckpt_path = ''

    return p
