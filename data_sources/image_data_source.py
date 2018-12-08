import os
import pickle
import numpy as np
from data_sources.data_source import DataSource

class ImageDataSource(DataSource):
    """
    A base class for an image data source. An image data source differs from a normal data source
    in that the whole dataset cannot be loaded into memory. Thus we work with pickle files instead,
    only loading data as needed.
    """
    def __init__(self, params):
        self.p = params
        self.data_tags = None
        self.training_dataset = None
        self.validation_dataset = None

        self.num_training_samples = self.p.trainer.batch_size * \
                                    int((self.p.trainer.training_set_size * self.p.trainer.num_samples) //
                                     self.p.trainer.batch_size)
        self.num_validation_samples = self.p.trainer.num_samples - self.num_training_samples
        
        # Controls whether get_data_tags returns
        # tags which reference pickle files
        # or tags for the data iself
        self.get_placeholder_data_tags = False
        self.actual_data_tags = None
        self.placeholder_data_tags = ['file_number_n1', 'num_samples_n1']

    def _create_image_dataset(self):
        # Get the old file list
        data_files = self.get_file_list()
        old_data_dir = self.p.data_creation.data_dir

        # Create a new directory for image data
        self.p.data_creation.data_dir = self._create_tmp_image_dir()

        # Initialize the simulator to render images
        simulator = self.p.simulator_params.simulator(self.p.simulator_params)

        for data_file in data_files:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)

            # Get the filename 'file{:d}.pkl' and file_number '{:d}'
            filename, fil_number = self._extract_file_name_and_number(data_file, old_data_dir)
            
            # Render the images from the simulator
            img_nmkd = self._render_image(simulator, data)

            # Save the image augmented data
            # to the new data_creation.data_dir
            img_filename = os.path.join(self.p.data_creation.data_dir, filename)
            data['img_nmkd'] = np.array(img_nmkd)
            with open(img_filename, 'wb') as f:
                pickle.dump(data, f)
            # TODO: Delete the tmp_dir later

    def _extract_file_name_and_number(self, filename, data_dir):
        """
        take a filename './dir0/dir1/.../dirn/file{:d}.pkl'
        and return:
            filename = 'file{:d}.pkl'
            file_number = {:d}
        """

        filename = os.path.relpath(filename, data_dir)  # file{:d}.pkl
        file_number = filename.split('.')[0].split('file')[-1]  # '{:d}'
        file_number = int(file_number)  # {:d}
        return filename, file_number

    def _render_image(self, simulator, data):
        """
        Uses the simulator to render the image the
        robot would have seen based on robot configurations
        in data.
        """
        if simulator.name == 'Circular_Obstacle_Map_Simulator':
            import pdb; pdb.set_trace()
            # TODO: get the occupancy grid from somewhere!!!
            img_nmkd = simulator.get_observation(pos_n3=data['vehicle_state_nk3'][:, 0],
                                                 obs_centers_nl2=data['obs_centers_nm2'],
                                                 obs_radii_nl1=data['obs_radii_nm1'],
                                                 occupancy_grid_positions_ego_1mk12=self.occupancy_grid_positions_ego_1mk12)
        elif simulator.name == 'SBPD_Simulator':
            img_nmkd = simulator.get_observation(pos_n3=data['vehicle_state_nk3'][:, 0],
                                                 crop_size=self.p.model.num_inputs.occupancy_grid_size)
        else:
            raise NotImplementedError
        return img_nmkd

    def _create_tmp_image_dir(self):
        """
        Create a temporary directory where image data
        can be saved.
        """
        from utils import utils 
        import datetime

        # Create a temporary directory for image data
        img_dir = 'tmp_{:s}_image_data_{:s}'.format(self.p.simulator_params.obstacle_map_params.renderer_params.camera_params.modalities[0],
                                                    datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
        img_dir = os.path.join(self.p.data_creation.data_dir, img_dir)
        utils.mkdir_if_missing(img_dir)
        return img_dir

    def load_dataset(self):
        """
        Augment the imageless data directory
        with images and save it in a temporary
        data directory. Load this new dataset.
        """

        # Render images for the imageless data
        # and save the new dataset in a temporary
        # directory used for training
        self._create_image_dataset()

        # Create dictionaries for train and validation sets
        # which store information such as the current pickle
        # file that is open, the current data that is loaded, etc.
        self._create_train_and_validation_information_dicts()

        # This allows for loading references to pickle files rather
        # than data itself
        self.get_placeholder_data_tags = True

        # Load the new dataset
        super().load_dataset()

        
    def get_data_tags(self, example_file, file_type='.pkl'):
        """
        Get the keys of the dictionary saved in the example file. 
        If get_placeholder_data_tags is true returns the placeholder tags
        else returns the actual_data_tags.
        """
        if file_type == '.pkl':
            # Save the actual data tags if they havent been already
            if self.actual_data_tags is None:
                super().get_data_tags(example_file, file_type)
                self.actual_data_tags = self.data_tags
            
            if self.get_placeholder_data_tags:
                self.data_tags = self.placeholder_data_tags
            else:
                self.data_tags = self.actual_data_tags
        else:
            raise NotImplementedError
 
    def _get_current_data(self, filename):
        """
        Load and return the data stored in filename.
        This can be overriden in subclasses
        (see image_data_source.py).
        """
        _, file_number = self._extract_file_name_and_number(filename,
                                                            self.p.data_creation.data_dir)
        with open(filename, 'rb') as handle:
            data_current = pickle.load(handle)
        n = self._get_n(data_current)
        data = {'file_number_n1': [[file_number]],
                'num_samples_n1': [[n]]}
        return data

    def _get_n(self, data):
        """
        Returns n, the batch size of the data inside
        this data dictionary.
        """
        raise NotImplementedError

    def _create_train_and_validation_information_dicts(self):
        """
        Create empty information dictionaries. These
        are used to persist information across batch loading.
        """
        def _create_info_dict():
            return {'filename': '',
                    'num_samples': 0,
                    'data': {}}

        self.training_info_dict = _create_info_dict()
        self.validation_info_dict = _create_info_dict()

    def generate_training_batch(self, start_index):
        """
        Generate a training batch from the dataset.
        """
        assert self.training_dataset is not None
        # Update self.data_tags to have the real data tags
        self.data_tags = self.actual_data_tags

        if start_index == 0:
            self.training_info_dict['num_samples'] = 0

        num_samples = np.cumsum(self.training_dataset['num_samples_n1'])
        file_idx = np.where(start_index < num_samples)[0][0]
        data_shuffle_idxs = self.training_shuffle_idxs[file_idx]

        n = self.training_dataset['num_samples_n1'][file_idx, 0]
       
        # Load the pickle file into memory if necessary
        self._load_data_into_info_dict(self.training_dataset, file_idx,
                                       self.training_info_dict)

        # Get the start index relative to the start of this pickle
        # files data
        start_index += -self.training_info_dict['num_samples'] + n

        # The whole batch can be loaded from one pickle file
        if start_index + self.p.trainer.batch_size < n:
            
            # Get the training batch
            training_batch = self.get_data_from_indices(self.training_info_dict['data'],
                                                        data_shuffle_idxs[start_index: start_index+self.p.trainer.batch_size])
        else:  # The batch is split over two pickle files
            # Get the remaining data from the first pickle file
            training_batch0 = self.get_data_from_indices(self.training_info_dict['data'],
                                                        data_shuffle_idxs[start_index: start_index+self.p.trainer.batch_size])

            remaining_num_samples = self.p.trainer.batch_size - self._get_n(training_batch0)
            
            # Load the next pickle file into memory
            file_idx += 1
            data_shuffle_idxs = self.training_shuffle_idxs[file_idx]
            self._load_data_into_info_dict(self.training_dataset, file_idx,
                                           self.training_info_dict)


            training_batch1 = self.get_data_from_indices(self.training_info_dict['data'],
                                                         data_shuffle_idxs[:remaining_num_samples])

            # Join the two sub-batches of data
            training_batch = {}
            for tag in self.actual_data_tags:
                training_batch[tag] = np.concatenate([training_batch0[tag], training_batch1[tag]],
                                                     axis=0)
        return training_batch
    
    def generate_validation_batch(self):
        """
        Generate a validation batch from the dataset.
        """
        assert self.validation_dataset is not None
        # Update self.data_tags to have the real data tags
        self.data_tags = self.actual_data_tags
        
        #TODO: This could be made faster. Dont need to load a new pickle file everytime!
        file_idx = np.random.choice(len(self.validation_dataset['file_number_n1'][:, 0]))
        data_shuffle_idxs = self.validation_shuffle_idxs[file_idx]
        
        # Load the pickle file into memory if necessary
        self._load_data_into_info_dict(self.validation_dataset, file_idx,
                                       self.validation_info_dict)
        
        # Sample indices and get data
        idxs = np.random.choice(len(data_shuffle_idxs), self.p.trainer.batch_size)
        validation_batch = self.get_data_from_indices(self.validation_info_dict['data'],
                                                      data_shuffle_idxs[idxs])
        return validation_batch

    def shuffle_datasets(self):
        """
        Shuffle the training and the validation datasets.
        This could be helpful in between the epochs for randomization.
        """
        # Update self.data_tags to have the placeholder data tags
        self.data_tags = self.placeholder_data_tags
        
        # Shuffle the order of the pickle files
        # for training and validation
        super().shuffle_datasets()

        # Generate indices which correspond to shuffling
        # individual examples inside pickle files
        self.training_shuffle_idxs = self._generate_shuffle_ind_for_data(self.training_dataset)
        self.validation_shuffle_idxs = self._generate_shuffle_ind_for_data(self.validation_dataset)

    def _load_data_into_info_dict(self, data, file_idx, info_dict):
        """
        Load the file in data at index file_idx into
        info_dict.
        """
        filename = 'file{:d}.pkl'.format(data['file_number_n1'][file_idx, 0])

        # Load the file if is it not already loaded
        if not filename == info_dict['filename']:
            info_dict['filename'] = filename

            filename = os.path.join(self.p.data_creation.data_dir, filename)
            with open(filename, 'rb') as f:
                data_current = pickle.load(f)

            info_dict['data'] = data_current
            info_dict['num_samples'] += data['num_samples_n1'][file_idx, 0]
        else:
            # If the file has already been loaded but num_samples has been reset to 0
            # increment it so that it correctly reflects the correct number
            if info_dict['num_samples'] == 0:
                info_dict['num_samples'] += data['num_samples_n1'][file_idx, 0]

    def _generate_shuffle_ind_for_data(self, data):
        """
        Generates shuffling data level shuffling arrays for each pickle
        file in data.
        """
        return [np.random.permutation(num_samples[0]) for num_samples in data['num_samples_n1']]

    def split_data_into_training_and_validation(self, data):
        """
        Split data intro training and validation sets.
        """
        # number of desired training and validation samples
        ts = self.num_training_samples
        vs = self.p.trainer.num_samples - self.num_training_samples

        # Find the file index for the training and validation sets
        idx_train = np.where(np.cumsum(data['num_samples_n1']) >= ts)[0][0] + 1
        idx_valid = np.where(np.cumsum(data['num_samples_n1'][idx_train:]) >= vs)[0][0] + 1
        idx_valid += idx_train

        training_dataset = {'file_number_n1': data['file_number_n1'][:idx_train],
                            'num_samples_n1': data['num_samples_n1'][:idx_train]}

        validation_dataset = {'file_number_n1': data['file_number_n1'][idx_train: idx_valid],
                              'num_samples_n1': data['num_samples_n1'][idx_train: idx_valid]}

        # Make sure there is enough training and validation data
        assert(np.sum(training_dataset['num_samples_n1']) >= ts)
        assert(np.sum(validation_dataset['num_samples_n1']) >= vs)

        return training_dataset, validation_dataset
