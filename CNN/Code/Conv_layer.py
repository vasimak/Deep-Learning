import numpy as np
from scipy import signal
from Dense_layer import *

class Convolution_Layer(Layer):
    def __init__(self, input_shape, kernel_size, depth):
        input_depth, input_height, input_width = input_shape
        self.depth = depth
        self.input_shape = input_shape
        self.input_depth = input_depth
        self.output_shape = (depth, input_height - kernel_size + 1, input_width - kernel_size + 1)
        self.kernels_shape = (depth, input_depth, kernel_size, kernel_size) # Το depth μας δίνει τον αριθμο των kernels, Το input depth τις διαστασεις του input, και το kernel size μας δινει το μεγεθος του πινακα στους kernel
        self.kernels = np.random.randn(*self.kernels_shape)                # Για παραδειγμα αν εχω ενα τρισδιαστατο input(3 καναλια), και εχω βαλει depth 5, και kernel sizei 3*3, θα εχω(5,3,3,3) 
        self.biases = np.random.randn(*self.output_shape)
        
    def forward(self, input):
        self.input = input
        self.output = np.copy(self.biases)
        for i in range(self.depth):
            for j in range(self.input_depth):
                self.output[i] += signal.correlate2d((self.input[j]), (self.kernels[i, j]), "valid")
        return self.output

    def backwards(self, d_output):
        self.d_kernels = np.zeros(self.kernels_shape)
        self.d_inputs= np.zeros(self.input_shape)

        for i in range(self.depth):
            for j in range(self.input_depth):
                self.d_kernels[i, j] = signal.correlate2d(self.input[j], d_output[i], "valid")
                self.d_inputs[j] += signal.convolve2d(d_output[i], self.kernels[i, j], "full")
        self.d_biases = np.sum(d_output,axis=0,keepdims=True)  
        # self.kernels -= learning_rate * self.d_kernels
        # self.biases -= learning_rate * self.d_biases 
 