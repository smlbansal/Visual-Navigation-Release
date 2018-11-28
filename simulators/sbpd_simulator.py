from obstacles.sbpd_map
from simulators.simulator import Simulator


class SBPDSimulator(Simulator):

    def __init__(self, params):
        assert(params.obstacle_map_params.obstacle_map is SBPDMap)
        super().__init__(params=params)

    def _reset_obstacle_map(self, rng):
        """
        For SBPD the obstacle map does not change
        between episodes.
        """
        return None

    def _update_fmm_map(self):
        """
        For SBPD the obstacle map and therefore
        the FMM Map does not change between
        episodes.
        """
        return None

    def _init_obstacle_map(self, rng):
        """ Initializes the sbpd map."""
        p = self.obstacle_map_params
        return p.obstacle_map(p)

    def _render_obstacle_map(self, ax):
        p = self.params
        raise NotImplementedError
