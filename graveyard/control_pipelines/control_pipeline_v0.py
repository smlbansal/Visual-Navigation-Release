from control_pipelines.control_pipeline import ControlPipeline


class Control_Pipeline_v0(ControlPipeline):
    """A control pipeline in which all trajectories in the pipeline are
    valid. """
    pipeline_name = 'v0'
    calculate_spline_speeds = False

    @staticmethod
    def keep_valid_problems(system_dynamics, k, planning_horizon_s,
                            start_config, goal_config, params):
        """All problems presented by start_config and goal_config
        are treated as valid in Control Pipeline V0."""
        return params.n
