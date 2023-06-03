import numpy as np

class Dropout_Layer:

    def __init__(self, rate):
        self.rate = 1 -rate

    def forward(self, inputs):

        self.inputs = inputs
        
        
        self.binary_mask = np.random.binomial(1,self.rate,size=inputs.shape) / self.rate

        self.output = inputs * self.binary_mask
    
    def backwards(self, d_output):

        self.d_inputs = d_output * self.binary_mask