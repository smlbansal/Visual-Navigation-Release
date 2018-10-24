import tensorflow as tf
from control_pipelines.control_pipeline import Control_Pipeline


class Control_Pipeline_v0(Control_Pipeline):
    """A control pipeline in which all trajectories in the pipeline are
    valid. """
    pipeline_name = 'v0'

    def _compute_valid_batch_idxs(self, horizon_s):
        """ All trajectories in the control pipeline are valid
        for pipeline v0"""
        self.valid_idxs = tf.range(self.params.n)
