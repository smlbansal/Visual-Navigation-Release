from control_pipelines.control_pipeline import ControlPipeline


class Control_Pipeline_v1(ControlPipeline):
    """ A control pipeline which masks off trajectories involving invalid splines.
    A valid spline is one that respects dynamic constraints on speed and
    angular speed within a given horizon_s."""
    pipeline_name = 'v1'

    def __init__(self, system_dynamics, params, precompute=False,
                 load_from_pickle_file=True, bin_velocity=True, v0=None,
                 k=None):
        super().__init__(system_dynamics, params, precompute, load_from_pickle_file,
                         bin_velocity, v0, k)
        self.calculate_spline_speeds = True

    def _compute_valid_batch_idxs(self, horizon_s):
        """ Computes the batch indices of the valid splines."""
        return self.traj_spline.check_dynamic_feasability(self.system_dynamics.v_bounds[1],
                                                          self.system_dynamics.w_bounds[1],
                                                          horizon_s=horizon_s)
