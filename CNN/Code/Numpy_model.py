
import numpy as np
from Conv_layer import *
from Softmax_function import *
from Loss_function import *
from Dropout_layer import *
from Dense_layer import *
from Optimizer_function import *
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import time as time


plot_filename = "plot.png"
model_output = "model.txt"

# Preprocess Function
data_limit = 100
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


# Load mnist and preprocess data
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X_train, y_train = preprocess_data(x_train, y_train)
X_test, y_test = preprocess_data(x_test, y_test)

# Initalize layers and functions
dropout_dense_1 = Dropout_Layer(0.5)
dropout_dense_2 = Dropout_Layer(0.2)
conv_1 = Convolution_Layer((1, 28, 28), 3, 10)
flatten=Reshape((10, 26, 26), (10 * 26 * 26, 1))
dense_1 = Layer_Dense(10 * 26* 26,128)
dense_2=Layer_Dense(128,10)
activation_conv_1 = Activation_ReLU()
activation_dense_1 = Activation_ReLU()
activation_dense_2 = Activation_ReLU()
softmax = Activation_Softmax()

loss_function = Loss_CategoricalCrossentropy()
# optimizer = Optimizer_Adam(learning_rate=10e-5,decay=10e-6)
optimizer = Optimizer_SGD(learning_rate=10e-5,decay=0)

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []




with open(model_output, "w") as f:
    for epoch in tqdm(range(41)):
        softmax.predictions=[]
        start_time = time.time()
        for x, y in tqdm(zip(X_train, y_train), desc=f'Epoch {epoch}'):
            
            # Forward pass
            conv_1.forward(x)
            activation_conv_1.forward(conv_1.output)
            flatten.forward(activation_conv_1.output)
            dense_1.forward(flatten.output)
            activation_dense_1.forward(dense_1.output)
            dropout_dense_1.forward(activation_dense_1.output)
            dense_2.forward(dropout_dense_1.output)
            activation_dense_2.forward(dense_2.output)
            dropout_dense_2.forward(activation_dense_2.output)
            softmax.forward(dropout_dense_2.output)
            loss_function.forward(softmax.output, y)
            
            # Backward pass
            loss_function.backwards(softmax.output, y)
            softmax.backwards(loss_function.d_inputs)
            dropout_dense_2.backwards(softmax.d_inputs)
            activation_dense_2.backwards(dropout_dense_2.d_inputs)
            dense_2.backwards(activation_dense_2.d_inputs)
            dropout_dense_1.backwards(dense_2.d_inputs)
            activation_dense_1.backwards(dropout_dense_1.d_inputs)
            dense_1.backwards(activation_dense_1.d_inputs)
            flatten.backwards(dense_1.d_inputs)
            activation_conv_1.backwards(flatten.d_inputs)
            conv_1.backwards(activation_conv_1.d_inputs)

            # Optimize gradients
            optimizer.pre_update_parameters()
            optimizer.update_parameters(conv_1)
            optimizer.update_parameters(dense_1)
            optimizer.update_parameters(dense_2)
            optimizer.post_update_parameters()

        # Training accuracy calculation
        y_reshaped = y_train.reshape(data_limit*10,10)
        predictions_array =np.asarray(softmax.predictions)
        predictions_reshaped = predictions_array.reshape(y_reshaped.shape)
        predictions_reshaped_true= np.argmax(predictions_reshaped,axis=1)
        if len(y_reshaped.shape) == 2:
            train_labels=np.argmax(y_reshaped,axis=1)  
        train_predictions = predictions_reshaped_true.reshape(train_labels.shape)
        training_accuracy = np.mean(train_predictions==train_labels)
        train_accuracies.append(training_accuracy)
        train_accuracies.append(training_accuracy)

        #Training loss calculation
        loss = loss_function.calculate(predictions_reshaped, y_reshaped)
        train_losses.append(loss)
        

        # Test predictions
        softmax.predictions = []
        test_losses_epoch = []
        test_predictions = []
        for x, y in zip(X_test, y_test):

            # Forward pass
            conv_1.forward(x)
            activation_conv_1.forward(conv_1.output)
            flatten.forward(activation_conv_1.output)
            dense_1.forward(flatten.output)
            activation_dense_1.forward(dense_1.output)
            dropout_dense_1.forward(activation_dense_1.output)
            dense_2.forward(dropout_dense_1.output)
            activation_dense_2.forward(dense_2.output)
            dropout_dense_2.forward(activation_dense_2.output)
            softmax.forward(dropout_dense_2.output)
            loss_function.forward(softmax.output, y)

            
        # Calculate test accuracy and loss
        test_predictions = np.asarray(softmax.predictions)
        test_predictions = test_predictions.reshape(len(test_predictions), -1)
        test_labels = y_test.reshape(len(y_test), -1)
        test_loss = loss_function.calculate(test_predictions, test_labels)
        test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == np.argmax(test_labels, axis=1))
        test_losses_epoch.append(test_loss)
        test_losses.append(np.mean(test_losses_epoch))
        test_accuracies.append(test_accuracy)
        
        if not epoch % 10:
            end_time = time.time()
            elapsed_time = end_time - start_time
            output = (f"Epoch {epoch} took {elapsed_time:.2f} seconds, "
                        f"training loss: {loss:.3f}, "
                        f"training accuracy: {training_accuracy:.3f}, "
                        f"test loss: {test_loss:.3f}, "
                        f"test accuracy: {test_accuracy:.3f}, "
                        f"optimizer: {optimizer.current_learning_rate}\n")
            print(output, end="")
            
            f.write(output)
        


# Plot training and test loss and accuracy
def save_plot(filename,train_losses,test_losses,train_accuracies,test_accuracies):
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Training')
    plt.plot(test_losses, label='Test')
    plt.title('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(train_accuracies, label='Training')
    plt.plot(test_accuracies, label='Test')
    plt.title('Accuracy');
    plt.xlabel('Epoch')
    plt.legend()
    plt.savefig(filename)
    plt.show()

save_plot(plot_filename,train_losses, test_losses,train_accuracies,test_accuracies)