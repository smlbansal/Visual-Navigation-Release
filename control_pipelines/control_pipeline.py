class Control_Pipeline:
    """A class representing a control pipeline.
    Used for planning trajectories between start and goal states
    """

    def plan(self, start_state, goal_state):
        """ Use the control pipeline to plan
        a trajectory from start_state to goal_state
        """
        raise NotImplementedError
