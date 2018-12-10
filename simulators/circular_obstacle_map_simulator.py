from obstacles.circular_obstacle_map import CircularObstacleMap
from models.visual_navigation.top_view.top_view_model import TopViewModel
from simulators.simulator import Simulator


class CircularObstacleMapSimulator(Simulator):
    name = 'Circular_Obstacle_Map_Simulator'

    def __init__(self, params):
        assert(params.obstacle_map_params.obstacle_map is CircularObstacleMap)
        super().__init__(params=params)

    def get_observation(self, config=None, pos_n3=None, **kwargs):
        """
        Return the robot's observation from configuration config
        or pos_nk3.
        """
        return self.obstacle_map.get_observation(config=config, pos_n3=pos_n3, **kwargs)

    def get_observation_from_data_dict_and_model_params(self, data_dict, model_params):
        """
        Returns the robot's observation from the data inside data_dict,
        using parameters specified by the model.
        """
        occupancy_grid_positions_ego_1mk12 = TopViewModel.initialize_occupancy_grid(model_params)
        img_nmkd = self.get_observation(pos_n3=data_dict['vehicle_state_nk3'][:, 0],
                                        obs_centers_nl2=data_dict['obs_centers_nm2'],
                                        obs_radii_nl1=data_dict['obs_radii_nm1'],
                                        occupancy_grid_positions_ego_1mk12=occupancy_grid_positions_ego_1mk12)
        return img_nmkd

    def _reset_obstacle_map(self, rng):
        p = self.params.reset_params.obstacle_map
        assert(p.reset_type == 'random')
        self.obstacle_map = self._init_obstacle_map(rng)

    def _update_fmm_map(self):
        """
        Update the fmm goal position map. For the circular obstacle
        map the underlying obstacle map may change so
        create a new fmm_map object.
        """
        self.fmm_map = self._init_fmm_map()
        self._update_obj_fn()

    def _init_obstacle_map(self, rng):
        """ Initializes a new circular obstacle map."""
        p = self.params
        return p.obstacle_map_params.obstacle_map.init_random_map(map_bounds=p.obstacle_map_params.map_bounds,
                                                                  rng=rng,
                                                                  reset_params=p.reset_params.obstacle_map.params,
                                                                  params=p.obstacle_map_params)

    def _render_obstacle_map(self, ax):
        p = self.params
        self.obstacle_map.render_with_obstacle_margins(ax,
                                                       margin0=p.avoid_obstacle_objective.obstacle_margin0,
                                                       margin1=p.avoid_obstacle_objective.obstacle_margin1)
