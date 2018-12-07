from params.top_view_trainer_params import create_params as create_params_top_view
from waypoint_grids.projected_image_space_grid import ProjectedImageSpaceGrid


def create_params():
    p = create_params_top_view()

    # Waypoint parameters
    p.simulator_params.planner_params.control_pipeline_params.waypoint_params.grid = ProjectedImageSpaceGrid
    
    # Data creation parameters
    p.data_creation.data_points = int(100e3)
    p.data_creation.data_points_per_file = 1000
    p.data_creation.data_dir = '/home/ext_drive/somilb/data/training_data/circular_obstacle_map/projected_topview_full_episode_100k'


    # Training parameters
    p.trainer.num_samples = int(20e3)
    p.trainer.num_epochs = 400

    # Checkpoint
    p.trainer.ckpt_path = '/home/vtolani/Documents/Projects/visual_mpc/logs/tmp/session_2018-12-06_14-51-30/checkpoints/ckpt-10'

    # Test params
    p.test.simulate_expert = True
    return p
