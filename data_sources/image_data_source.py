import os
import pickle
import numpy as np
from data_sources.data_source import DataSource


class ImageDataSource(DataSource):
    """
    A base class for an image data source. An image data source differs from a normal data source
    in that the whole dataset cannot be loaded into memory. 
    
    When generating data, an image data source still generates image-less data. When loading data
    (i.e. for training) the image data augments the imageless dataset with images (if needed), saving it in a
    new directory. Since the entire dataset will not fit in memory an image_data_source stores the
    training and validation sets as dictionaries with references to pickle files (placeholders). Upon calling
    generate_training_batch (or validation_batch) the references to pickle files are converted into
    actual training data (by loading the actual pickle file).
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
        """
        Load the image-less data in the given data_dir, augment
        this dataset with images, and save the resulting image dataset
        in a new directory. If the data already exists, do nothing.
        """

        # Get the old file list
        data_files = self.get_file_list()
        old_data_dir = self.p.data_creation.data_dir

        # Create a new directory for image data
        self.p.data_creation.data_dir = self._create_image_dir()

        # If the image data already exists, no need to recreate it
        if len(os.listdir(self.p.data_creation.data_dir)) > 0:
            return self.p.data_creation.data_dir

        # Initialize the simulator and model to render images
        simulator = self.p.simulator_params.simulator(self.p.simulator_params)

        for data_file in data_files:
            with open(data_file, 'rb') as f:
                data = pickle.load(f)

            # Get the filename 'file{:d}.pkl' and file_number '{:d}'
            filename, _ = self._extract_file_name_and_number(data_file, old_data_dir)

            # Render the images from the simulator
            img_nmkd = simulator.get_observation_from_data_dict_and_model(data,
                                                                          self.model)

            # Save the image augmented data
            # to the new data_creation.data_dir
            img_filename = os.path.join(self.p.data_creation.data_dir, filename)
            data['img_nmkd'] = np.array(img_nmkd)
            with open(img_filename, 'wb') as f:
                pickle.dump(data, f)

        return self.p.data_creation.data_dir

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

    def _create_image_dir(self):
        """
        Create a new directory where image data
        can be saved.
        """
        from utils import utils 
        
        img_dir = self._get_image_dir_name()
        img_dir = os.path.join(self.p.data_creation.data_dir, img_dir)
        utils.mkdir_if_missing(img_dir)
        return img_dir

    def _get_image_dir_name(self):
        """
        Return the name of a unique directory
        where image data can be saved.
        """
        raise NotImplementedError

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

        # Set this flag to True so that the data that is passed around
        # is a reference to a pickle file (memory efficient)
        # rather than actual data (memory inefficient)
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
        An image data_source deals with references to data
        files, only loading as needed.
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
        are used to persist information (keep a particular data file loaded)
        across batches.
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

        # Get the index of the current data file to use
        num_samples = np.cumsum(self.training_dataset['num_samples_n1'])
        file_idx = np.where(start_index < num_samples)[0][0]

        # Get the indices used to shuffle the data inside this data file
        data_shuffle_idxs = self.training_shuffle_idxs[file_idx]

        # Number of samples in this data file
        n = self.training_dataset['num_samples_n1'][file_idx, 0]

        # Load the pickle file into memory if necessary
        self._load_data_into_info_dict(self.training_dataset, file_idx,
                                       self.training_info_dict)

        # Get the start index relative to the start of this data_files data
        start_index += -self.training_info_dict['num_samples'] + n

        # The whole batch can be loaded from one data file
        if start_index + self.p.trainer.batch_size < n:
            
            # Get the training batch
            training_batch = self.get_data_from_indices(self.training_info_dict['data'],
                                                        data_shuffle_idxs[start_index: start_index+self.p.trainer.batch_size])
        else:  # The batch is split over two data_files
            
            # Get the remaining data from the first data_file
            training_batch0 = self.get_data_from_indices(self.training_info_dict['data'],
                                                         data_shuffle_idxs[start_index: start_index+self.p.trainer.batch_size])

            remaining_num_samples = self.p.trainer.batch_size - self._get_n(training_batch0)
           
            # The index of the next data file
            file_idx += 1

            # The indices used to shuffle the data inside this file
            data_shuffle_idxs = self.training_shuffle_idxs[file_idx]

            # Load the next data_file into memory
            self._load_data_into_info_dict(self.training_dataset, file_idx,
                                           self.training_info_dict)

            # Get the rest of the batch from the second data_file
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
        
        # TODO: This could be made faster. This could potentially
        # load a new pickle file everytime it is called which may slow
        # things down
        
        # Choose a random data_file from the validation set
        file_idx = np.random.choice(len(self.validation_dataset['file_number_n1'][:, 0]))

        # Get the shuffling indices for this data_file
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
        
        # Shuffle the order of the data_files used in training and validation
        super().shuffle_datasets()

        # Generate indices which correspond to shuffling
        # individual examples inside data_files
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
            # increment it so that it correctly reflects the correct number of samples
            if info_dict['num_samples'] == 0:
                info_dict['num_samples'] += data['num_samples_n1'][file_idx, 0]

    def _generate_shuffle_ind_for_data(self, data):
        """
        Generates shuffling arrays for the actual data points inside each data
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
