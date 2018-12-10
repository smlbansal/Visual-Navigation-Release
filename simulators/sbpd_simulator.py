from obstacles.sbpd_map import SBPDMap
from simulators.simulator import Simulator


class SBPDSimulator(Simulator):
    name = 'SBPD_Simulator'

    def __init__(self, params):
        assert(params.obstacle_map_params.obstacle_map is SBPDMap)
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
        img_nmkd = self.get_observation(pos_n3=data_dict['vehicle_state_nk3'][:, 0],
                                        crop_size=model_params.num_inputs.occupancy_grid_size)
        return img_nmkd

    def _reset_obstacle_map(self, rng):
        """
        For SBPD the obstacle map does not change
        between episodes.
        """
        return False

    def _update_fmm_map(self):
        """
        For SBPD the obstacle map does not change,
        so just update the goal position.
        """
        if hasattr(self, 'fmm_map'):
            goal_pos_n2 = self.goal_config.position_nk2()[:, 0]
            self.fmm_map.change_goal(goal_pos_n2)
        else:
            self.fmm_map = self._init_fmm_map()
        self._update_obj_fn()

    def _init_obstacle_map(self, rng):
        """ Initializes the sbpd map."""
        p = self.params.obstacle_map_params
        return p.obstacle_map(p)

    def _render_obstacle_map(self, ax):
        p = self.params
        self.obstacle_map.render_with_obstacle_margins(ax, start_config=self.start_config,
                                                       margin0=p.avoid_obstacle_objective.obstacle_margin0,
                                                       margin1=p.avoid_obstacle_objective.obstacle_margin1)
