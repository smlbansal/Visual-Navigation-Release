from params.rgb_trainer.sbpd.uniform_grid.resnet50.rgb_trainer_finetune_params import create_params as create_rgb_trainer_params
from dotmap import DotMap


def create_params():
    p = create_rgb_trainer_params()

    # Change the number of inputs to the model
    p.model.num_outputs = 3  # (x, y ,theta)
    
    # Image size to [224, 224, 3]
    p.model.num_inputs.image_size = [224, 224, 3]
    
    # Finetune the resnet weights
    p.model.arch.finetune_resnet_weights = True

    # Change the learning rate and num_samples
    p.trainer.lr = 1e-4
    p.trainer.num_samples = int(150e3)
    
    # Checkpoint settings
    p.trainer.ckpt_save_frequency = 1
    p.trainer.restore_from_ckpt = True

    # Change the number of tests and callback frequency
    p.trainer.callback_frequency = 500
    p.trainer.callback_number_tests = 200

    # Change the Data Processing parameters
    p.data_processing.input_processing_function = 'resnet50_keras_preprocessing'

    # Input processing parameters
    p.data_processing.input_processing_params = DotMap(
        p=0.1,  # Probability of distortion
        version=''  # Version of the distortion function
    )

    # Change the data_dir
    # Projected Grid
    p.data_creation.data_dir = [
        '/home/ext_drive/somilb/data/training_data/sbpd/sbpd_projected_grid/area3/full_episode_random_v1_100k',
        '/home/ext_drive/somilb/data/training_data/sbpd/sbpd_projected_grid/area4/full_episode_random_v1_100k',
        '/home/ext_drive/somilb/data/training_data/sbpd/sbpd_projected_grid/area5a/full_episode_random_v1_100k']
    
    # # Uniform Grid
    # p.data_creation.data_dir = [
    #     '/home/ext_drive/somilb/data/training_data/sbpd/uniform_grid/area3/full_episode_random_v1_100k',
    #     '/home/ext_drive/somilb/data/training_data/sbpd/uniform_grid/area4/full_episode_random_v1_100k',
    #     '/home/ext_drive/somilb/data/training_data/sbpd/uniform_grid/area5a/full_episode_random_v1_100k']

    # Change the checkpoint
    p.trainer.ckpt_path = '/home/somilb/Documents/Projects/visual_mpc/tmp/session_2019-01-16_17-57-05/' \
                          'checkpoints/ckpt-5'

    # Seed for selecting the test scenarios and the number of such scenarios
    p.test.seed = 10
    p.test.number_tests = 200
    
    # Let's not look at the expert
    p.test.simulate_expert = False
    
    # Only use the valid goals
    p.test.expert_success_goals.use = True
    p.test.expert_success_goals.dirname = '/home/ext_drive/somilb/data/expert_data/sbpd/uniform_grid'
    
    # Parameters for the metric curves
    p.test.metric_curves = DotMap(start_ckpt=1,
                                  end_ckpt=10,
                                  start_seed=1,
                                  end_seed=10,
                                  plot_curves=True
                                  )

    # from tensorflow.contrib.memory_stats.python.ops.memory_stats_ops import BytesInUse
    # with tf.device('/device:GPU:1'):  # Replace with device you are interested in
    #     bytes_in_use = BytesInUse()
    # print(bytes_in_use / (1024 * 1024), 'before')
    
    return p
