from params.top_view_trainer.top_view_trainer_params import create_params as create_top_view_trainer_params
from training_utils.data_processing.rgb_preprocess import preprocess as preprocess_image_data


def create_params():
    p = create_top_view_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 60  # (v, omega) for 30 timesteps

    # Change the learning rate
    p.trainer.lr = 1e-5

    return p
