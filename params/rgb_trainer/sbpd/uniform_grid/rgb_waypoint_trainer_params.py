from params.rgb_trainer.sbpd.uniform_grid.rgb_trainer_params import create_params as create_rgb_trainer_params
from dotmap import DotMap


def create_params():
    p = create_rgb_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 3  # (x, y ,theta)

    # Change the learning rate and num_samples
    p.trainer.lr = 1e-4
    p.trainer.num_samples = int(100e3)
    
    # Change the number of tests
    p.trainer.callback_number_tests = 200

    # Change the Data Processing parameters
    p.data_processing.input_processing_function = 'normalize_images'
    
    # Input processing parameters
    p.data_processing.input_processing_params = DotMap(
                                                        p=0.1  # Probability of distortion
                                                      )

    # Change the data_dir
    p.data_creation.data_dir = ['/home/ext_drive/somilb/data/training_data/sbpd/uniform_grid/area3/full_episode_random_v1_100k',
                                '/home/ext_drive/somilb/data/training_data/sbpd/uniform_grid/area4/full_episode_random_v1_100k',
                                '/home/ext_drive/somilb/data/training_data/sbpd/uniform_grid/area5a/full_episode_random_v1_100k']

    # Change the checkpoint
    p.trainer.ckpt_path = '/home/somilb/Documents/Projects/visual_mpc/tmp/custom_arch/session_2019-01-09_13-38-30/checkpoints/ckpt-5'

    # Seed for selecting the test scenarios and the number of such scenarios
    p.test.seed = 10
    p.test.number_tests = 100

    # Parameters for the metric curves
    p.test.metric_curves = DotMap(start_ckpt=1,
                                  end_ckpt=17,
                                  start_seed=1,
                                  end_seed=10,
                                  plot_curves=True
                                  )

    return p
