from obstacles.sbpd_map import SBPDMap
from simulators.simulator import Simulator


class SBPDSimulator(Simulator):
    name = 'SBPD_Simulator'

    def __init__(self, params):
        assert(params.obstacle_map_params.obstacle_map is SBPDMap)
        super().__init__(params=params)

    def get_observation(self, config):
        """
        Return the robot's observation from configuration config.
        """
        return self.obstacle_map.get_observation(config)

    def _reset_obstacle_map(self, rng):
        """
        For SBPD the obstacle map does not change
        between episodes.
        """
        return None

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
        self.obstacle_map.render(ax, start_config=self.start_config)
