import tensorflow as tf
import numpy as np
from planners.planner import Planner
from trajectory.trajectory import Trajectory, SystemConfig


class NNWaypointPlanner(Planner):
    """ A planner which selects an optimal waypoint using
    a trained neural network. """

    def optimize(self, start_config):
        """ Optimize the objective over a trajectory
        starting from start_config.
        """
        p = self.params

        model = p.model
        
        # TODO: need access to the simulator here
        raw_data = # create data

        processed_data = model.create_nn_inputs_and_outputs(raw_data)
        
        # Predict the NN output
        nn_output = model.predict_nn_output(processed_data['inputs'], is_training=False)
        

        obj_vals, data = self.eval_objective(start_config, nn_output)
        min_idx = tf.argmin(obj_vals)
        min_cost = obj_vals[min_idx]

        waypts, horizons_s, trajectories, controllers = data

        self.opt_waypt.assign_from_config_batch_idx(waypts, min_idx)
        self.opt_traj.assign_from_trajectory_batch_idx(trajectories, min_idx)

        # Convert horizon in seconds to horizon in # of steps
        min_horizon = int(tf.ceil(horizons_s[min_idx, 0]/self.params.dt).numpy())
        min_controllers = {'K_1kfd': controllers['K_nkfd'][min_idx:min_idx+1],
                           'k_1kf1': controllers['k_nkf1'][min_idx:min_idx+1]}

        return self.opt_waypt, self.opt_traj, min_cost, min_horizon, min_controllers
