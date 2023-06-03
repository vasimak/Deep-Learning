import numpy as np

import matplotlib.pyplot as plt


class Layer_Dense:
    #Initialization of Layers
    def __init__(self,n_inputs,n_neurons):
        self.weights = 0.01* np.random.randn(n_inputs, n_neurons) 
        self.biases = np.zeros((1,n_neurons))

    #Forward pass
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights) + self.biases
    
    def backwards(self, d_output,learning_rate=1):
        self.d_weights = np.dot(self.inputs.T,d_output)

        self.d_biases = np.sum(d_output,axis=0,keepdims=True)   

        self.d_inputs = np.dot(d_output,self.weights.T)
    
        self.weights -= learning_rate * self.d_weights
        # print(self.weights)
        self.biases -= learning_rate * self.d_biases

class BinaryCrossEntropyLoss:
    def __init__(self):
        self.y_true = None
        self.y_pred = None

    def forward(self, y_true, y_pred):
        self.y_true = y_true
        self.y_pred = np.clip(y_pred, 1e-7, 1 - 1e-7)  # Clip predictions to avoid numerical instability
        loss = -np.mean(self.y_true * np.log(self.y_pred) + (1 - self.y_true) * np.log(1 - self.y_pred))
        return loss

    def backward(self,grad_output):
        batch_size = self.y_true.shape[0]
        self.d_inputs = grad_output * (self.y_pred - self.y_true) / batch_size

class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        print(self.output)
        return self.output

    def backward(self, grad):
        self.d_inputs = grad * self.output * (1 - self.output)

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



X= [[0, 0],[0, 1],[1, 0], [1, 1]]
y = [[0],[1], [1],[0]]
X =np.array(X)
y = np.array(y)
data = np.concatenate((X, y), axis=1)

# Shuffle the data
np.random.shuffle(data)

# Split the shuffled data back into X and y
X = data[:, :2]
y = data[:, 2:]


dense = Layer_Dense(2,1)
activation = Sigmoid()
optimizer = Optimizer_SGD(learning_rate=0.2,decay=0,momentum=0)
loss = BinaryCrossEntropyLoss()

for epoch in range(21):
    dense.forward(X)
    activation.forward(dense.output)
    loss.forward(activation.output,y)
    loss.backward(activation.output)
    activation.backward(loss.d_inputs)
    dense.backwards(activation.d_inputs)

   
    predictions = np.argmax(activation.output,axis=1)
    # print(predictions)
    if len(y.shape) == 2:
        y_true=y.T
        y_true1 = np.ravel(y_true)
    accuracy = np.mean(predictions==y_true1)

    
    if not epoch % 10:
        print(f'epoch: {epoch}, ' +
            # f'loss: {loss:.3f}, ' +
            f'accuracy: {accuracy:.3f}, ' +
            f'optimizer: {optimizer.current_learning_rate}')