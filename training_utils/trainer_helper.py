import tensorflow as tf


class TrainerHelper(object):
    
    def __init__(self, params):
        self.p = params
        
    def train(self, model, data_source, callback_fn=None):
        """
        Train a given model.
        """
        # Create the optimizer
        self.optimizer = self.create_optimizer()
        
        # Compute the total number of training samples
        num_training_samples = self.p.batch_size * int((self.p.training_size * self.p.num_samples) // self.p.batch_size)
        
        # Keep a track of the performance
        epoch_performance_training = []
        epoch_performance_validation = []
        
        # Begin the training
        for i in range(self.p.num_epochs):
            # Shuffle the dataset
            data_source.shuffle_datasets()

            # Define the metrics to keep a track of average loss over the epoch.
            # TODO(Somil, Varun): Define the average loss metrics.
            training_loss_metric = None
            validation_loss_metric = None
            
            # For loop over the training samples
            for j in range(0, num_training_samples, self.p.batch_size):
                # Get a training and a validation batch
                training_batch = data_source.generate_training_batch(j)
                validation_batch = data_source.generate_validation_batch()
                
                # Compute the loss and gradients
                with tf.GradientTape() as tape:
                    loss = model.compute_loss_function(training_batch, is_training=True, return_loss_components=False)
                
                # Take an optimization step
                grads = tape.gradient(loss, model.get_traiable_vars())
                self.optimizer.apply_gradients(zip(grads, model.get_traiable_vars()),
                                               global_step=tf.train.get_or_create_global_step())
                
                # Record the average loss for the training and the validation batch
                self.record_average_loss_for_batch(model, training_batch, validation_batch, training_loss_metric,
                                                   validation_loss_metric)
                
            # Do all the things required at the end of epochs including saving the checkpoints
            self.finish_epoch_processing(callback_fn)
            
            
    def load_checkpoint(self, ckpt):
        """
        Load a given checkpoint.
        """
        raise NotImplementedError
    
    def save_checkpoint(self):
        """
        Create and save a checkpoint.
        """
        raise NotImplementedError
    
    def create_optimizer(self):
        """
        Create an optimizer for the training. All changes in the learning rate should be handled here.
        """
        raise NotImplementedError
    
    def record_average_loss_for_batch(self, model, training_batch, validation_batch, training_loss_metric,
                                      validation_loss_metric):
        """
        Record the average loss for the batch and update the metric.
        """
        _, prediction_loss_training, _ = model.compute_loss_function(training_batch, is_training=False,
                                                                     return_loss_components=True)
        _, prediction_loss_validation, _ = model.compute_loss_function(validation_batch, is_training=False,
                                                                       return_loss_components=True)
        # Now add the loss values to the metric aggregation
        raise NotImplementedError

    def finish_epoch_processing(self, callback_fn=None):
        """
        Finish the epoch processing for example recording the average epoch loss for the training and the validation
        sets, save the checkpoint, hand over the control to the callback function etc.
        """
        raise NotImplementedError
