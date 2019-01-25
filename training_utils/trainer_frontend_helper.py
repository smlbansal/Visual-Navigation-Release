import tensorflow as tf
import numpy as np
import argparse
import os
import importlib
import datetime
import logging

from data_sources.data_source import DataSource
from training_utils.trainer_helper import TrainerHelper
from models.base import BaseModel
from utils import utils, log_utils


class TrainerFrontendHelper(object):
    """
    A base class for setting up a data collector, trainer or test.
    Exampple: to run a trainer file:
    PYTHONPATH='.' python executable/top_view_trainer.py generate-data --job-dir ./tmp/test
    --params params/top_view_trainer_params.py --d 0
    
    """
    def run(self):
        tf.enable_eager_execution(**utils.tf_session_config())
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
        self.create_session_dir(args)
        
        # Configure the device
        if args.device == -1:
            self.p.device = '/cpu:0'
        else:
            self.p.device = '/gpu:%i' % args.device

        # Parse parameters
        self.parse_params(self.p, args)

        # Setup the logger and dump the parameters
        self.setup_logger_and_dump_params(args)

        # Configure plotting
        utils.configure_plotting()
        
        # Run the command
        if args.command == 'generate-data':
            self.generate_data()
        elif args.command == 'train':
            self.train()
        elif args.command == 'test':
            self.test()
        elif args.command == 'generate-metric-curves':
            self.generate_metric_curves()
        else:
            raise NotImplementedError('Unknown command')
 
    def parse_params(self, p, args):
        """
        Parse the parameters to add some additional
        helpful information.
        """
        return p

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
        self.trainer = TrainerHelper(self.p)
    
    def generate_data(self, params=None):
        """
        Generate the data using the data source.
        """
        with tf.device(self.p.device):
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
            # Create an input and output model
            self.create_model()

            # Create a data source and load the data
            self.create_data_source()
            self.data_source.load_dataset()

            # Create a trainer
            self.create_trainer()
            
            # Maybe restore a checkpoint
            self.maybe_restore_checkpoint()

            # Start the training
            self.trainer.train(model=self.model, data_source=self.data_source,
                               callback_fn=self.callback_fn)

    def callback_fn(self, lcl):
        """
        A callback function that is called after a training epoch.
        lcl is a key, value mapping of the current state of the local
        variables in the trainer.
        """
        return None

    def test(self):
        """
        Test a trained network.
        """
        # Set the random seed
        if self.p.test.seed != -1:
            np.random.seed(seed=self.p.test.seed)
            tf.set_random_seed(seed=self.p.test.seed)

        # Load the checkpoint
        with tf.device(self.p.device):
            # Create an input and output model
            self.create_model()
    
            # Create a trainer
            self.create_trainer()
    
            # Load the checkpoint
            self.trainer.restore_checkpoint(model=self.model)

    def create_params(self, param_file):
        """
        Create the parameters given the path of the parameter file.
        """
        # Execute this if python > 3.4
        try:
            spec = importlib.util.spec_from_file_location('parameter_loader', param_file)
            foo = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(foo)
        except AttributeError:
            # Execute this if python = 2.7 (i.e. when running on real robot with ROS)
            module_name = param_file.replace('/', '.').replace('.py', '')
            foo = importlib.import_module(module_name)
        return foo.create_params() 
        
    def create_session_dir(self, args):
        """
        Create the job and the session directories.
        """
        # Store the test data with the data
        # of the trained network you are testing
        if args.command == 'test':
            trainer_dir = self.p.trainer.ckpt_path.split('checkpoints')[0]
            checkpoint_number = int(self.p.trainer.ckpt_path.split('checkpoints')[1].split('-')[1])
            job_dir = os.path.join(trainer_dir, 'test', 'checkpoint_{:d}'.format(checkpoint_number))
        else:
            job_dir = args.job_dir

        # Create the job directory if required
        utils.mkdir_if_missing(job_dir)
        self.p.job_dir = job_dir

        # Create the session directory
        self.p.session_dir = os.path.join(self.p.job_dir,
                                          'session_%s' % (datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')))
        os.mkdir(self.p.session_dir)

    def setup_logger_and_dump_params(self, args):
        """
        Dump all the paramaters in params.json in the session directory, and then setup a logger.
        """
        # Create a parameter json file
        utils.log_dict_as_json(self.p, os.path.join(self.p.session_dir, 'params.json'))
        
        # Setup a logger
        # TODO(Somil, Varun): This is a hack for now. Maybe make it more sophisticated.
        # TODO: Put this back
        #log_utils.setup_logger(filename=os.path.join(self.p.session_dir, 'log.txt'))
        
        # Add some basic information to the logger
        #logging.info('Parameter file name: %s' % args.params)
        #logging.info('Command: %s' % args.command)
        
    def generate_metric_curves(self):
        """
        Generate the metric curve from the starting checkpoint to end checkpoint over the range of specified seeds.
        """
        raise NotImplementedError
    
    def maybe_restore_checkpoint(self):
        """
        Optionally restore a checkpoint and start training from there
        """
        if self.p.trainer.restore_from_ckpt:
            # Restore the checkpoint
            self.trainer.restore_checkpoint(model=self.model)
            
            # TODO(Somil, Varun): In future, we may wanna save the results in the same session directory in which the
            # checkpoint currently resides. In that case, we should also set the correct epoch and checkpoint numbers
            # for saving results.
        

if __name__ == '__main__':
    TrainerFrontendHelper().run()
