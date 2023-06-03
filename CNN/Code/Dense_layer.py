import numpy as np

class Layer_Dense:
    #Initialization of Layers
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.01* np.random.randn(n_inputs, n_neurons) 
        self.biases = np.zeros((1,n_neurons))

    #Forward pass
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights) + self.biases
    
    def backwards(self, d_output):
        self.d_weights = np.dot(self.inputs.T,d_output)

        self.d_biases = np.sum(d_output,axis=0,keepdims=True)   

        self.d_inputs = np.dot(d_output,self.weights.T)
        # self.weights -= learning_rate * self.d_weights
        # self.biases -= learning_rate * self.d_biases
        

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient):
        # TODO: update parameters and return input gradient
        pass



class Activation_ReLU:
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.maximum(0,inputs)
    
    def backwards(self, d_output):
        self.d_inputs = d_output.copy()

        self.d_inputs[self.inputs<=0] = 0
     


    def predictions(self,outputs):
        return outputs

class Reshape(Layer):
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape
        self.output_shape = output_shape

    def forward(self, input):
        self.output = np.reshape(input, self.output_shape)
        self.output= self.output.T

    def backwards(self, output_gradient):
        self.d_inputs = np.reshape(output_gradient, self.input_shape)