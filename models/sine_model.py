from models.base import BaseModel


class SineModel(BaseModel):
    
    def create_nn_inputs_and_outputs(self, raw_data):
        return raw_data
