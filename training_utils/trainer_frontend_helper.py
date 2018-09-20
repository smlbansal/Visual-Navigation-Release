import tensorflow as tf
import numpy as np
import argparse
import os
import importlib
import datetime

from data_sources.data_source import DataSource
from training_utils.trainer_helper import TrainerHelper
from models.base import BaseModel


class TrainerFrontendHelper(object):
    """
    A base class for setting up a data collector, trainer or test.
    """
    def run(self):
        tf.enable_eager_execution()
        self.setup_parser()
    
    def setup_parser(self):
        """
        Parse the command line inputs and run the appropriate function. Also, create the session directory.
        """
        parser = argparse.ArgumentParser(description='Process the command line inputs')
        parser.add_argument("command", help='the command to run')
        parser.add_argument("-j", "--job-dir", required=True,
                            help='the path to the job directory to save the output of the session')
        parser.add_argument("-p", "--params", required=True, help='the path to the parameter file')
        parser.add_argument("-d", "--device", type=int, default=1, help='the device to run the training/test on')
        
        args = parser.parse_args()
        
        self.configure_parser(args)
    
    def configure_parser(self, args):
        """
        Configure the arguments to take the appropraite actions, and add them to parameter list.
        """
        # Create the parameters
        assert os.path.exists(args.params)
        self.p = self.create_params(args.params)

        # Create the job and session directories
        self.create_session_dir(args.job_dir)
        
        # Configure the device
        if args.device == 0:
            self.p.device = '/cpu:0'
        else:
            self.p.device = '/gpu:%i' % args.device
        
        # Run the command
        if args.command == 'generate-data':
            self.generate_data()
        elif args.command == 'train':
            self.train()
        elif args.command == 'test':
            self.test()
        else:
            raise NotImplementedError('Unknown command')
        
    def create_data_source(self, params=None):
        """
        Create a data source for the data manipulation.
        """
        self.data_source = DataSource(self.p)
        
    def create_model(self, params=None):
        """
        Create the input-output model.
        """
        self.model = BaseModel(self.p)
    
    def create_trainer(self, params=None):
        """
        Create a trainer for training.
        """
        self.trainer = TrainerHelper(self.p.trainer)
    
    def generate_data(self, params=None):
        """
        Generate the data using the data source.
        """
        self.create_data_source(params)
        self.data_source.generate_data()
    
    def train(self):
        """
        Start a trainer and begin training.
        """
        # Set the random seed
        if self.p.trainer.seed != -1:
            np.random.seed(seed=self.p.trainer.seed)
            tf.set_random_seed(seed=self.p.trainer.seed)
        
        # Start the training
        with tf.device(self.p.device):
            # First create a data source and load the data
            self.create_data_source()
            self.data_source.load_dataset()
            
            # Create an input and output model
            self.create_model()
            
            # Create a trainer
            self.create_trainer()
            
            # Start the training
            self.trainer.train(model=self.model, data_source=self.data_source)

    def test(self, params):
        """
        Test a trained network.
        """
        raise NotImplementedError('Should be implemented by the child class')
    
    def create_params(self, param_file):
        """
        Create the parameters given the path of the parameter file.
        """
        spec = importlib.util.spec_from_file_location('parameter_loader', param_file)
        foo = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(foo)
        return foo.create_params()
        
    def create_session_dir(self, job_dir):
        """
        Create the job and the session directories.
        """
        # Create the job directory if required
        if not os.path.exists(job_dir):
            os.makedirs(job_dir)
        self.p.job_dir = job_dir

        # Create the session directory
        self.p.session_dir = os.path.join(self.p.job_dir,
                                          'session_%s' % (datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
        os.mkdir(self.p.session_dir)
    

if __name__ == '__main__':
    TrainerFrontendHelper().run()
