from control_pipelines.control_pipeline import ControlPipeline


class Control_Pipeline_v1(ControlPipeline):
    """ A control pipeline which masks off trajectories involving invalid splines.
    A valid spline is one that respects dynamic constraints on speed and
    angular speed within a given horizon_s."""
    pipeline_name = 'v1'

    def _compute_valid_batch_idxs(self, horizon_s):
        """ Computes the batch indices of the valid splines."""
        self.valid_idxs = self.traj_spline.check_dynamic_feasability(self.system_dynamics.v_bounds[1],
                                                                     self.system_dynamics.w_bounds[1],
                                                                     horizon_s=horizon_s)
