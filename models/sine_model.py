from models.base import BaseModel


class SineModel(BaseModel):
    
    def create_nn_inputs_and_outputs(self, raw_data, is_training=None):
        return raw_data
