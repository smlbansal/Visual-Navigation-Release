import numpy as np
import tensorflow as tf
from trajectory.trajectory import SystemConfig
from obstacles.turtlebot_map import TurtlebotMap
from simulators.simulator import Simulator


class TurtlebotSimulator(Simulator):
    name = 'Turtlebot_Simulator'

    def __init__(self, params):
        assert(params.obstacle_map_params.obstacle_map is TurtlebotMap)
        super(TurtlebotSimulator, self).__init__(params=params)

    # TODO Varun T.: this is a hack to make the turtlebot work for now.
    # Change the control pipeline, planner, simulator strcuture so that
    # this is not needed (more info in TOD0 in control_pipeline_v0.plan)
    @staticmethod
    def parse_params(p):
        """
        Parse the parameters to add some additional helpful parameters.
        """
        p = super(TurtlebotSimulator, TurtlebotSimulator).parse_params(p)

        # Pass the control horizon to the control pipeline. Normally
        # control pipeline simulates controls for length planning_horizon,
        # however when running on the real robot we only want to simulate
        # controls for control_horizon
        p.planner_params.control_pipeline_params.control_horizon = p.control_horizon
        p.planner_params.planning_horizon = p.control_horizon
        return p

    def simulate(self):
        self.system_dynamics.hardware.track_states = True
        super(TurtlebotSimulator, self).simulate()
        self.system_dynamics.hardware.track_states = False

    def _init_obj_fn(self):
        """
        A dummy objective function which always returns 0.0
        so the robot works with the simulator structure.
        """
        from dotmap import DotMap
        dummy_obj = DotMap()
        dummy_obj.evaluate_function = lambda trajectories: [0.0]
        return dummy_obj

    def get_observation(self, config=None, pos_n3=None, **kwargs):
        """
        Return the robot's observation from configuration config
        or pos_nk3.
        """
        return self.obstacle_map.get_observation(config=config, pos_n3=pos_n3, **kwargs)

    def _reset_start_configuration(self, rng):
        """
        Reset the Turtlebot's odometer.
        """
        p = self.params
        self.system_dynamics.hardware.reset_odom()
        # ideally should be 0, but when using gazebo it will be some small nonzero number
        assert(np.linalg.norm(np.array(self.system_dynamics.hardware.state)*1.) < 1e-3)
        self.start_config = self.system_dynamics.init_egocentric_robot_config(dt=p.dt,
                                                                              n=1)

    def _reset_goal_configuration(self, rng):
        """
        Reset the turtlebot goal position
        """
        p = self.params

        assert(p.reset_params.goal_config.position.reset_type == 'custom')
        x, y = p.reset_params.goal_config.position.goal_pos
        pos_112 = np.array([[[x, y]]], dtype=np.float32)
        self.goal_config = SystemConfig(dt=p.dt, n=1, k=1,
                                        position_nk2=pos_112)
        return False

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

    def _dist_to_goal(self, trajectory):
        """
        Calculate the l2 distance to the goal.
        """
        dist_to_goal_1k = tf.norm(trajectory.position_nk2() - self.goal_config.position_nk2(),
                                  axis=2)
        return dist_to_goal_1k

    def _compute_time_idx_for_collision(self, vehicle_trajectory):
        if self.system_dynamics.hardware.hit_obstacle:
            import pdb; pdb.set_trace()
        else:
            time_idx = tf.constant(self.params.episode_horizon+1)
        return time_idx

    def _render_obstacle_map(self, ax):
        """
        The obstacle map is not known a priori so
        do nothing.
        """
        return None
