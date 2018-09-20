import os
import pickle
import numpy as np

from data_sources.data_source import DataSource


class SineDataSource(DataSource):

    def generate_data(self):
        # Create the data directory if required
        if not os.path.exists(self.p.data_creation.data_dir):
            os.makedirs(self.p.data_creation.data_dir)
        
        # Generate the data
        counter = 1
        for _ in range(0, self.p.data_creation.data_points, self.p.data_creation.data_points_per_file):
            x = np.random.uniform(-10., 10., (self.p.data_creation.data_points_per_file, 1)).astype(np.float32)
            sinx = np.sin(x)
            
            # Create a data dictionary
            data = {'inputs': x, 'labels': sinx}

            # Save the file
            filename = os.path.join(self.p.data_creation.data_dir, 'file%i.pkl' % counter)
            with open(filename, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            counter += 1

    def load_dataset(self):
        # Get all the files in the directory
        file_list = self.get_file_list()
        
        # Concatenate the data corresponding to a list of files
        data = self.concatenate_file_data(file_list)
        
        # Shuffle the data and create the training and the validation datasets
        data = self.shuffle_data_dictionary(data)
        self.training_dataset = self.get_data_from_indices(data, np.arange(self.num_training_samples))
        self.validation_dataset = self.get_data_from_indices(data, np.arange(self.num_training_samples,
                                                                             self.p.trainer.num_samples))
