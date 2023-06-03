import numpy as np


class Activation_Softmax:
    def forward(self,inputs):
        self.inputs = inputs
        
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities

    def backwards(self,d_output):
        self.d_inputs = np.empty_like(d_output)
        
        for index, (single_output,single_d_output) in enumerate(zip(self.output,d_output)):
            single_output = single_output.reshape(-1,1)

            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output,single_output.T)

            self.d_inputs[index] = np.dot(jacobian_matrix,single_d_output)
        # n =np.size(self.output)
        # sel.d_inputs= np.dot((np.identity(n) - self.output.T) * self.output, output_gradient)

    def predictions(self,output):
        return np.argmax(output,axis=1)