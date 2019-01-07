from obstacles.turtlebot_map import TurtlebotMap
from simulators.simulator import Simulator


class TurtlebotSimulator(Simulator):
    name = 'Turtlebot_Simulator'

    def __init__(self, params):
        assert(params.obstacle_map_params.obstacle_map is TurtlebotMap)
        super().__init__(params=params)

    def get_observation(self, config=None, pos_n3=None, **kwargs):
        """
        Return the robot's observation from configuration config
        or pos_nk3.
        """
        return self.obstacle_map.get_observation(config=config, pos_n3=pos_n3, **kwargs)


    def _reset_start_configuration(self):
        """
        Reset the Turtlebot's odometer.
        """
        self.system_dynamics.hardware.reset_odom()
        import pdb; pdb.set_trace()

    def _reset_goal_configuration(self):
        """
        Reset the turtlebot goal position
        """
        assert(self.params.reset_params.goal_config.reset_type == 'custom')
        import pdb; pdb.set_trace()
        return True


    def _reset_obstacle_map(self, rng):
        """
        For Turtlebot the obstacle map holds
        no environment information and thus
        does not need to be reset
        """
        return False

    def _update_fmm_map(self):
        """
        For Turtlebot there is no fmm map.
        """
        return False

    def _init_obstacle_map(self, rng):
        """ Initializes the sbpd map."""
        p = self.params.obstacle_map_params
        return p.obstacle_map(p)

    def _render_obstacle_map(self, ax):
        """
        The obstacle map is not known a priori so
        do nothing.
        """
        return None
