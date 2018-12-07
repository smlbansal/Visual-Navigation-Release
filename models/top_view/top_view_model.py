import tensorflow as tf

from models.base import BaseModel
from training_utils.architecture.simple_cnn import simple_cnn


class TopViewModel(BaseModel):
    
    def __init__(self, params):
        super(TopViewModel, self).__init__(params=params)
        
        # Initialize an empty occupancy grid
        self.initialize_occupancy_grid()

        # TODO: Potentially set this elsewhere so the model can use different
        # simulators (i.e. for different SBPD areas)
        # Instantiate a simulator object used by the model to generate images
        self._simulator = params.simulator_params.simulator(params.simulator_params)

    def make_architecture(self):
        """
        Create the CNN architecture for the model.
        """
        self.arch = simple_cnn(image_size=self.p.model.num_inputs.occupancy_grid_size,
                               num_inputs=self.p.model.num_inputs.num_state_features,
                               num_outputs=self.p.model.num_outputs,
                               params=self.p.model.arch)
    
    def initialize_occupancy_grid(self):
        """
        Create an empty occupancy grid for training and test purposes.
        """
        x_size = self.p.model.occupancy_grid_dx[0] * self.p.model.num_inputs.occupancy_grid_size[0]
        y_size = 0.5 * self.p.model.occupancy_grid_dx[1] * self.p.model.num_inputs.occupancy_grid_size[1]
        
        x_k = tf.linspace(0., 1., self.p.model.num_inputs.occupancy_grid_size[0]) * x_size
        y_m = tf.linspace(1., -1., self.p.model.num_inputs.occupancy_grid_size[1]) * y_size
        xx_mk, yy_mk = tf.meshgrid(x_k, y_m, indexing='xy')
        
        self.occupancy_grid_positions_ego_1mk12 = tf.stack([xx_mk, yy_mk], axis=2)[tf.newaxis, :, :, tf.newaxis, :]

    def create_occupancy_grid(self, raw_data):
        """
        Create an occupancy grid of size m x k around the current vehicle position.
        """
        return raw_data['img_nmkd']
        #if self._simulator.name == 'Circular_Obstacle_Map_Simulator':
        #    grid_nmk1 = self._simulator.get_observation(pos_n3=raw_data['vehicle_state_nk3'][:, 0],
        #                                                obs_centers_nl2=raw_data['obs_centers_nm2'],
        #                                                obs_radii_nl1=raw_data['obs_radii_nm1'],
        #                                                occupancy_grid_positions_ego_1mk12=self.occupancy_grid_positions_ego_1mk12)
        #elif self._simulator.name == 'SBPD_Simulator':
        #    grid_nmk1 = self._simulator.get_observation(pos_n3=raw_data['vehicle_state_nk3'][:, 0],
        #                                                crop_size=self.p.model.num_inputs.occupancy_grid_size)
        #else:
        #    raise NotImplementedError
        #return grid_nmk1
