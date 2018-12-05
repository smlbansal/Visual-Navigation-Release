from params.top_view_trainer_params import create_params as create_params_top_view
from waypoint_grids.projected_image_space_grid import ProjectedImageSpaceGrid


def create_params():
    p = create_params_top_view()

    # Waypoint parameters
    p.simulator_params.planner_params.control_pipeline_params.waypoint_params.grid = ProjectedImageSpaceGrid
    
    # Data creation parameters
    p.data_creation.data_points = int(1000)
    p.data_creation.data_points_per_file = 100
    p.data_creation.data_dir = '/home/ext_drive/somilb/data/projected_topview_full_episode_100k'
    
    return p
