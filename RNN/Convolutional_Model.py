import numpy as np
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import time


data_limit = 100
plot_filename = "plot.png"
model_output = "model.txt"
def preprocess_data(x, y):
    x_new = []
    y_new = []
    for i in range(10):
        indices = np.where(y == i)[0][:data_limit]
        np.random.shuffle(indices)  # Shuffle the indices before selecting data
        x_new.append(x[indices])
        y_new.append(np.full(len(indices), i))
    x_new = np.concatenate(x_new)
    y_new = np.concatenate(y_new)
    x_new = x_new.reshape(len(x_new), 1, 28, 28)
    x_new = x_new.astype("float32") / 255
    y_new = np_utils.to_categorical(y_new)
    y_new = y_new.reshape(len(y_new), 10, 1)
    indices = np.random.permutation(len(x_new))  # Shuffle the data
    return x_new[indices], y_new[indices]


# load MNIST from server, limit to 100 images per class since we're not training on GPU
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X_train, y_train = preprocess_data(x_train, y_train)
X_test, y_test = preprocess_data(x_test, y_test)


network = [
    Convolution_Layer((1, 28, 28), 3, 10),
    Activation_ReLU(),
    Reshape((10, 26, 26), (10 * 26 * 26, 1)),
    Layer_Dense(10 * 26 * 26, 128),
    Activation_ReLU(),
    Dropout_Layer(0.2),
    Layer_Dense(128, 10),
    Activation_ReLU(),
    Dropout_Layer(0.2),
    Activation_Softmax(),
    Loss_CategoricalCrossentropy()
 
]

train(
    network,X_train, y_train,X_test,y_test, epochs = 51
)