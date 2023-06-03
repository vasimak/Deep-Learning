import numpy as np
from softmax import *
from loss import *
from optimizer import *
from dropout import *
import matplotlib.pyplot as plt


np.random.seed(0)
X =[[1,2,3,2.5],
    [2,5,-1,2],
    [1.5,2.7,3.3,-0.8]]


#temporary dataset
def spiral_data(points, classes):
    X = np.zeros((points*classes, 2))
    y = np.zeros(points*classes, dtype='uint8')
    for class_number in range(classes):
        ix = range(points*class_number, points*(class_number+1))
        r = np.linspace(0.0, 1, points)  # radius
        t = np.linspace(class_number*4, (class_number+1)*4, points) + np.random.randn(points)*0.2
        X[ix] = np.c_[r*np.sin(t*2.5), r*np.cos(t*2.5)]
        y[ix] = class_number
    return X, y




class Layer_Dense:
    #Initialization of Layers
    def __init__(self,n_inputs,n_neurons, weights_regularizer_l1 = 0, weights_regularizer_l2 = 0, bias_regularizer_l1 = 0, bias_regularizer_l2 = 0):
        self.weights = 0.01* np.random.randn(n_inputs, n_neurons) 
        self.biases = np.zeros((1,n_neurons))

        self.weights_regularizer_l1 = weights_regularizer_l1
        self.weights_regularizer_l2 = weights_regularizer_l2
        self.bias_regularizer_l1 = bias_regularizer_l1
        self.bias_regularizer_l2 = bias_regularizer_l2
    #Forward pass
    def forward(self,inputs):
        self.inputs = inputs
        self.output = np.dot(inputs,self.weights) + self.biases
    
    def backwards(self, d_output):
        self.d_weights = np.dot(self.inputs.T,d_output)
        self.d_biases = np.sum(d_output,axis=0,keepdims=True)   

        if self.weights_regularizer_l1 > 0:
            dL1 = np.ones_like(self.weights)
            dL1[self.weights < 0] = -1
            self.d_weights += self.weights_regularizer_l1*dL1
        if self.weights_regularizer_l2 > 0:
            self.d_weights += 2*self.weights_regularizer_l2*self.d_weights
        if self.bias_regularizer_l1 > 0:
            dL1 = np.ones_like(self.biases)
            dL1[self.biases < 0] = -1
            self.d_biases += self.bias_regularizer_l1*dL1
        if self.bias_regularizer_l2 > 0:
            self.d_biases+= 2*self.bias_regularizer_l2*self.d_biases

        self.d_inputs = np.dot(d_output,self.weights.T)
        #         self.weights -= learning_rate * weights_gradient
        # self.bias -= learning_rate * output_gradient
        # return input_gradient

class Layer:
    def __init__(self):
        self.input = None
        self.output = None

    def forward(self, input):
        # TODO: return output
        pass

    def backward(self, output_gradient, learning_rate):
        # TODO: update parameters and return input gradient
        pass

class Layer_Input:
    def forward(self,inputs,training):
        self.output =inputs     

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

    def backward(self, output_gradient, learning_rate):
        self.d_inputs = np.reshape(output_gradient, self.input_shape)
# (X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()

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

print("Shuffled X:")
print(X)
print("Shuffled y:")
print(y)
print(X.shape, y.shape)
dropout1 = Dropout_Layer(0.1)
dense1 = Layer_Dense(2,2)
activation1 = Activation_ReLU()
activation2 = Activation_Softmax()
loss_function= Loss_CategoricalCrossentropy()
optimizer = Optimizer_SGD(learning_rate=0.2,decay=0,momentum=0)
loss_function = Loss_CategoricalCrossentropy()
def binary_cross_entropy(y_true, y_pred):
    epsilon = 1e-7  # small value to prevent division by zero
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)  # clip predictions to avoid numerical instability
    loss = -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))
    return loss
# optimizer = Optimizer_Adam(learning_rate=0.02,decay=0.0001)
for epoch in range(1):
    dense1.forward(X)
    sigmoid = 1 / (1 + np.exp(-dense1.output))
   
    loss = loss_function.calculate(sigmoid,y)
   
    # print(sigmoid)
    print(y)
    predictions = np.argmax(sigmoid,axis=1)
    print(predictions)
    if len(y.shape) == 2:
        y_true=y.T
        y_true1 = np.ravel(y_true)
    accuracy = np.mean(predictions==y_true1)
    print(accuracy)
    print(y_true1,predictions.shape)
    loss_function.backwards(sigmoid, y)
    dense1.backwards(loss_function.d_inputs)
  
    optimizer.pre_update_parameters()
    optimizer.update_parameters(dense1)
    optimizer.post_update_parameters()

    if not epoch % 10:
        print(f'epoch: {epoch}, ' +
            f'loss: {loss:.3f}, ' +
            f'accuracy: {accuracy:.3f}, ' +
            f'optimizer: {optimizer.current_learning_rate}')
    

   