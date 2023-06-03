import numpy as np 

class Optimizer_SGD:

    def __init__(self,learning_rate=1.0, decay = 0, momentum = 0):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.momentum = momentum

    def pre_update_parameters(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. /(1. + self.decay * self.iterations))

    def update_parameters(self,layer):
        if self.momentum:

            if not hasattr(layer, 'weight_momentums'):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)

            weight_updates = self.momentum * layer.weight_momentums - self.current_learning_rate * layer.d_weights
            layer.weight_momentums = weight_updates

            bias_updates = self.momentum * layer.bias_momentums - self.current_learning_rate * layer.d_biases
            layer.bias_momentums =bias_updates
        else :
            weight_updates= -self.current_learning_rate * layer.d_weights
            bias_updates = -self.current_learning_rate * layer.d_biases

        layer.weights += weight_updates
        layer.biases += bias_updates
    def post_update_parameters(self):
        self.iterations+=1


class Optimizer_Adam:

    def __init__(self, learning_rate=1.0, decay = 0, epsilon = 1e-7, beta1=0.9, beta2=0.999):
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.decay = decay
        self.iterations = 0
        self.epsilon = epsilon
        self.beta1 = beta1
        self.beta2 = beta2

    def pre_update_parameters(self):
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1. /(1. + self.decay * self.iterations))

    def update_parameters(self,layer):

        if not hasattr(layer, 'weight_cache'):
            layer.weight_cache = np.zeros_like(layer.weights)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_cache = np.zeros_like(layer.biases)
            layer.bias_momentums = np.zeros_like(layer.biases)

        layer.weight_momentums = self.beta1 * layer.weight_momentums + (1 - self.beta1) * layer.d_weights
        layer.bias_momentums = self.beta1 * layer.bias_momentums + (1 - self.beta1) * layer.d_biases

        weight_momentums_corrected = layer.weight_momentums / (1 - self.beta1 ** (self.iterations +1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta1 ** (self.iterations +1))

        layer.weight_cache = self.beta2 * layer.weight_cache + (1 - self.beta2) * layer.d_weights**2
        layer.bias_cache = self.beta2 * layer.bias_cache + (1 - self.beta2) * layer.d_biases**2

        weight_cache_corrected = layer.weight_cache / (1 - self.beta2 ** (self.iterations+1))
        bias_cache_corrected = layer.bias_cache / (1 - self.beta2 ** (self.iterations+1))

        layer.weights += -self.current_learning_rate * weight_momentums_corrected / (np.sqrt(weight_cache_corrected) + self.epsilon)
        layer.biases += -self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cache_corrected) + self.epsilon)

    def post_update_parameters(self):
        self.iterations+=1
