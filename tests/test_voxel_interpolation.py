import numpy as np
from scipy import interpolate
import tensorflow as tf
from utils.voxel_map_utils import VoxelMap


def test_voxel_interpolation():
    # Create a grid and a function within that grid
    scale = 0.1
    grid_size = np.array([31, 31])
    grid_origin = np.array([5., 6.])

    x = np.linspace(grid_origin[0], grid_origin[0] + scale * (grid_size[0]-1), grid_size[0])
    y = np.linspace(grid_origin[1], grid_origin[1] + scale * (grid_size[1]-1), grid_size[1])
  
    xv, yv = np.meshgrid(x, y, sparse=False, indexing='xy')
    zv = 3*xv + 5*yv
  
    voxel_map = VoxelMap(scale=scale,
                         origin_2=tf.constant(grid_origin/scale, dtype=tf.float32),
                         map_size_2=tf.constant(grid_size, dtype=tf.float32),
                         function_array_mn=tf.constant(zv, dtype=tf.float32))
  
    # Let's have a bunch of points to test the interpolation
    test_positions = tf.constant([[[5.02, 6.01], [5.67, 7.73], [6.93, 6.93]], [[9.1, 7.2], [7.889, 8.22], [7.1, 8.1]]],
                                 dtype=tf.float32)
  
    # Get the interpolated output of the voxel map
    interpolated_values = voxel_map.compute_voxel_function(test_positions, invalid_value=-1.)
    interpolated_values = tf.reshape(interpolated_values, [-1]).numpy()
  
    # Expected interpolated values
    expected_interpolated_values = 3*test_positions[:, :, 0] + 5*test_positions[:, :, 1]
    expected_interpolated_values = tf.reshape(expected_interpolated_values, [-1]).numpy()
    expected_interpolated_values[3] = -1.
  
    # Scipy Interpolated values
    f_scipy = interpolate.RectBivariateSpline(y, x, zv, kx=1, ky=1)
    scipy_interpolated_values = f_scipy.ev(tf.reshape(test_positions[:, :, 1], [-1]).numpy(),
                                           tf.reshape(test_positions[:, :, 0], [-1]).numpy())
    scipy_interpolated_values[3] = -1.
  
    assert np.sum(abs(expected_interpolated_values - interpolated_values) <= 0.01) == 6
    assert np.sum(abs(scipy_interpolated_values - interpolated_values) <= 0.01) == 6


if __name__ == '__main__':
    tf.enable_eager_execution()
    np.random.seed(seed=1)
    test_voxel_interpolation()
