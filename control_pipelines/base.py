import os
import pickle
import numpy as np


class ControlPipelineBase(object):
    """A parent class representing an abstract control pipeline. It defines the basic functions that a control pipeline
    should expose. A control pipeline is used for planning trajectories between start and waypoint/goal configs.
    """

    def __init__(self, params):
        self.params = params.pipeline.parse_params(params)
        self.system_dynamics = params.system_dynamics_params.system(dt=params.system_dynamics_params.dt,
                                                                    params=params.system_dynamics_params)
        self.pipeline_files = self.valid_file_names()

    @staticmethod
    def parse_params(p):
        """
        Parse the parameters to add some additional helpful parameters.
        """
        # Parse the dependencies
        p.waypoint_params.grid.parse_params(p.waypoint_params)
        p.planning_horizon_s = p.spline_params.max_final_time
        p.planning_horizon = int(np.ceil(p.planning_horizon_s / p.system_dynamics_params.dt))
        return p

    @classmethod
    def get_pipeline(cls, params):
        """Used to instantiate a control pipeline. Can be overidden to ensure that only one pipeline is ever
        created (see pipeline v0)."""
        return cls(params)

    def generate_control_pipeline(self, params=None):
        """
        Generate a control pipeline from scratch using the generate_parameters, and potentially using some additional
        params. This function should generate the state and control trajectories, as well as the corresponding feedback
        controllers and horizon, and save them.
        """
        raise NotImplementedError

    def load_control_pipeline(self, params=None):
        """
        Load a control pipeline if it exists, otherwise throws an error.
        """
        if self.does_pipeline_exist():
            self._load_control_pipeline(params=params)
        else:
            assert(False,
                   'Control pipeline does not exist! Generate the pipeline first.')

    def _load_control_pipeline(self, params=None):
        """
        An internal function to load the pipeline that should be implemented by the child class. This function should
        load all the files in self.pipeline_files and then create data structures that are subsequently used.
        """
        raise NotImplementedError

    def save_control_pipeline(self, data, filename, file_format='.pkl'):
        """
        Save the data dictionary given by data in the file given by filename. This function should be typically called
        by generate_control_pipeline function.
        """
        if file_format == '.pkl':
            with open(filename, 'wb') as f:
                pickle.dump(data, f)
        else:
            raise NotImplementedError

    def does_pipeline_exist(self):
        """
        Check whether a pipeline exists already or not. This is a helpful function for any module that interact with
        a control pipeline object, such as planner.
        """
        return np.array([os.path.isfile(p) for p in self.pipeline_files]).all()

    def valid_file_names(self, file_format='.pkl'):
        """
        This function should return the list of the names of all files that a control pipeline should create.
        """
        raise NotImplementedError

    def plan(self, start_config):
        """Use the control pipeline to find the set of trajectories in the control pipeline that are suitable for
        start_config. The function should return list of valid_waypoints, corresponding horizons,
        state-control trajectories, and controllers. It is assumed that the pipeline exists and already has been
        loaded."""
        raise NotImplementedError

    # TODO: Fill this function in
    def render(self):
        raise NotImplementedError
