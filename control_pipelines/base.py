import os
import pickle


class ControlPipelineBase(object):
    """A parent class representing an abstract control pipeline.
    It defines the basic functions that a control pipeline should expose.
    
    A control pipeline is used for planning trajectories between start and goal configs.
    """

    def __init__(self, params):
        self.p = params
        self.pipeline_files = self.valid_file_names()
        
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
            assert(False, 'Control pipeline does not exist! Generate the pipeline first.')
    
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
        # ToDO(Varun): Implement this function in the parent class itself as this should be a generic function to all
        # child classes based on the parameters.
        raise NotImplementedError
    
    def does_pipeline_exist(self):
        """
        Check whether a pipeline exists already or not. This is a helpful function for any module that interact with
        a control pipeline object, such as planner.
        """
        # ToDO(Varun): Implement this function in the parent class itself. This function should check if self.p.dir has
        # all the files in self.pipeline_files.
        raise NotImplementedError
    
    def valid_file_names(self):
        """
        This function should return the list of the names of all files that a control pipeline should create.
        """
        raise NotImplementedError
    
    def plan(self, start_config, goal_config):
        """Use the control pipeline to find the set of trajectories in the control pipeline that are suitable for
        start_config and goal_config. The function should return list of valid_waypoints, corresponding horizons,
        state-control trajectories, and controllers. It is assumed that the pipeline exists and already has been
        loaded."""
        raise NotImplementedError
