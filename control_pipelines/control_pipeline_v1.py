import tensorflow as tf
from control_pipelines.control_pipeline import ControlPipeline


class Control_Pipeline_v1(ControlPipeline):
    """ A control pipeline which masks off trajectories involving invalid splines.
    A valid spline is one that respects dynamic constraints on speed and
    angular speed within a given horizon_s."""
    pipeline_name = 'v1'
    calculate_spline_speeds = True

    @staticmethod
    def keep_valid_problems(system_dynamics, k, planning_horizon_s,
                                start_config, goal_config, params):
        """Compute which problems are valid by constructing a spline
        from start_config to goal_config and checking dynamic feasability.
        Updates start_config and goal_config objects in place and returns
        n', the new batch size for this control pipeline."""
        traj_spline = params._spline(dt=params.dt, n=params.n, k=k,
                                params=params)
        ts_nk = tf.tile(tf.linspace(0., planning_horizon_s, k)[None], [params.n, 1])
        traj_spline.fit(start_config=start_config, goal_config=goal_config,
                             factors=None)
        traj_spline.eval_spline(ts_nk, calculate_speeds=True)
        valid_idxs = traj_spline.check_dynamic_feasability(system_dynamics.v_bounds[1],
                                                           system_dynamics.w_bounds[1],
                                                           horizon_s=planning_horizon_s)
        start_config.gather_across_batch_dim(valid_idxs)
        goal_config.gather_across_batch_dim(valid_idxs)
        return len(valid_idxs.numpy())
