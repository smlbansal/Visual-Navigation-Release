from params.top_view_trainer.top_view_trainer_params import create_params as create_trainer_params
from params.obstacle_map.sbpd_obstacle_map_params import create_params as create_sbpd_map_params


def create_params():
    p = create_trainer_params()

    # Change the input to the model
    p.model.num_inputs.image_size = [64, 64, 3]

    # Ensure the obstacle map is SBPD
    p.simulator_params.obstacle_map_params = create_sbpd_map_params()

    # Ensure the renderer modality is rgb
    p.simulator_params.obstacle_map_params.renderer_params.camera_params.modalities = ['rgb']

    return p
