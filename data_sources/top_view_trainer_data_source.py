import os
import pickle
import numpy as np

from data_sources.data_source import DataSource


class TopViewDataSource(DataSource):
    
    def generate_data(self):
        # Create the data directory if required
        if not os.path.exists(self.p.data_creation.data_dir):
            os.makedirs(self.p.data_creation.data_dir)
        
        # Generate the data
        counter = 1
        for _ in range(0, self.p.data_creation.data_points, self.p.data_creation.data_points_per_file):
            # Reset the data dictionary
            data = self.reset_data_dictionary()
            
            for _ in range(0, self.p.data_creation.data_points_per_file):
                # Reset the simulator
            
                # Run the planner for one step
                
                # Append the data to the current data dictionary
                
            # Prepare the dictionary for saving purposes
            
            # Save the data
            filename = os.path.join(self.p.data_creation.data_dir, 'file%i.pkl' % counter)
            with open(filename, 'wb') as handle:
                pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)
            
            # Increase the counter
            counter += 1
    
