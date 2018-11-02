from obstacles.circular_obstacle_map import CircularObstacleMap
from simulators.simulator import Simulator


class CircularObstacleMapSimulator(Simulator):

    def __init__(self, params):
        assert(params.obstacle_map_params.classname is CircularObstacleMap)
        super().__init__(params=params)

    def _reset_obstacle_map(self, rng):
        p = self.params.simulator_params.reset_params.obstacle_map
        assert(p.reset_type == 'random')
        self.obstacle_map = self._init_obstacle_map(rng, p.params)

    def _update_fmm_map(self):
        self.fmm_map = self._init_fmm_map()
        self._update_obj_fn()

    def _init_obstacle_map(self, rng):
        """ Initializes a new circular obstacle map."""
        p = self.params
        return p.obstacle_map_params.classname.init_random_map(map_bounds=p.obstacle_map_params.map_bounds,
                                                               rng=rng,
                                                               **p.reset_params.obstacle_map.params)

    def _render_obstacle_map(self, ax):
        p = self.params
        self.obstacle_map.render_with_obstacle_margins(ax,
                                                       margin0=p.avoid_obstacle_objective.obstacle_margin0,
                                                       margin1=p.avoid_obstacle_objective.obstacle_margin1)
