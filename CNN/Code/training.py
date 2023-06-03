import numpy as np
from layer_conv import *
from softmax_conv import *
from loss_conv import *
from dropout_conv import *
from conv import *
from optimizer_conv import *
# from rnn import *
import tensorflow as tf
from keras.datasets import mnist
from keras.utils import np_utils
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from datetime import datetime
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
# print(X_train.shape,y_train.shape)
dropout1 = Dropout_Layer(0.4)
dropout2 = Dropout_Layer(0.2)
dropout3 = Dropout_Layer(0.2)
dropout4 = Dropout_Layer(0.4)
conv1 = Convolution_Layer((1, 28, 28), 3, 10)
conv2 = Convolution_Layer((1, 26, 26), 3, 10)
r=Reshape((10, 26, 26), (10 * 26 * 26, 1))

dense1 = Layer_Dense(10*26*26,128)
activation1 = Activation_ReLU()
activation2 = Activation_ReLU()
activation3 = Activation_ReLU()
activation4 = Activation_ReLU()
softmax = Activation_Softmax()
dense2=Layer_Dense(128,10)
loss_function = Loss_CategoricalCrossentropy()
# optimizer = Optimizer_Adam(learning_rate=10e-5,decay=10e-6)
optimizer = Optimizer_SGD(learning_rate=10e-5,decay=0)

train_losses = []
train_accuracies = []
test_losses = []
test_accuracies = []




with open(model_output, "w") as f:
    for epoch in tqdm(range(21)):
        predictions=[]
        start_time = time.time()
        for x, y in tqdm(zip(X_train, y_train), desc=f'Epoch {epoch}'):
            
            conv1.forward(x)
            activation1.forward(conv1.output)
            r.forward(activation1.output)
            dense1.forward(r.output)
            activation2.forward(dense1.output)
            dropout2.forward(activation2.output)
            rnn1.forward(dropout2.output)
            activation4.forward(rnn1.output)
            dense2.forward(activation4.output)
            activation3.forward(dense2.output)
            dropout3.forward(activation3.output)
            softmax.forward(dropout3.output)
            predictions.append(softmax.output)
            loss_function.forward(softmax.output, y)
            
            
            loss_function.backwards(softmax.output, y)

            softmax.backwards(loss_function.d_inputs)
            dropout3.backwards(softmax.d_inputs)
            activation3.backwards(dropout3.d_inputs)
            dense2.backwards(activation3.d_inputs)
            activation4.backwards(dense2.d_inputs)
            rnn1.backward(activation4.d_inputs,10e-5)
            dropout2.backwards(rnn1.d_inputs)
            activation2.backwards(dropout2.d_inputs)
            dense1.backwards(activation2.d_inputs)
            r.backwards(dense1.d_inputs)
            activation1.backwards(r.d_inputs)
            conv1.backwards(activation1.d_inputs)
            optimizer.pre_update_parameters()
            optimizer.update_parameters(conv1)
            optimizer.update_parameters(dense1)
            optimizer.update_parameters(dense2)
            optimizer.post_update_parameters()

        y_reshaped = y_train.reshape(data_limit*10,10)
        predictions_array =np.asarray(predictions)
        predictions_reshaped = predictions_array.reshape(y_reshaped.shape)
        predictions_reshaped_true= np.argmax(predictions_reshaped,axis=1)

        if len(y_reshaped.shape) == 2:
            train_labels=np.argmax(y_reshaped,axis=1)  

        train_predictions = predictions_reshaped_true.reshape(train_labels.shape)
        # print(train_predictions.shape,train_labels.shape)
        accuracy = np.mean(train_predictions==train_labels)
        

        train_accuracies.append(accuracy)

        loss = loss_function.calculate(predictions_reshaped, y_reshaped)
        # loss=loss[0]
        train_losses.append(loss)
    

        # Test predictions
        test_losses_epoch = []
        test_predictions = []
        for x, y in zip(X_test, y_test):
            conv1.forward(x)
            activation1.forward(conv1.output)
            r.forward(activation1.output)
            
            dense1.forward(r.output)
            activation2.forward(dense1.output)
            dropout2.forward(activation2.output)
            dense2.forward(dropout2.output)
            activation3.forward(dense2.output)
            dropout3.forward(activation3.output)
            softmax.forward(dropout3.output)
            
         

            test_predictions.append(softmax.output)
        # Calculate test accuracy and loss
        test_predictions = np.asarray(test_predictions)
        test_predictions = test_predictions.reshape(len(test_predictions), -1)
        test_labels = y_test.reshape(len(y_test), -1)
        test_loss = loss_function.calculate(test_predictions, test_labels)
        test_accuracy = np.mean(np.argmax(test_predictions, axis=1) == np.argmax(test_labels, axis=1))
        test_losses_epoch.append(test_loss)

    
        if not epoch % 10:
            end_time = time.time()
            elapsed_time = end_time - start_time
            output = (f"Epoch {epoch} took {elapsed_time:.2f} seconds, "
                        f"training loss: {loss:.3f}, "
                        f"training accuracy: {accuracy:.3f}, "
                        f"test loss: {test_loss:.3f}, "
                        f"test accuracy: {test_accuracy:.3f}, "
                        f"optimizer: {optimizer.current_learning_rate}\n")
            print(output, end="")
            
            f.write(output)
        test_losses.append(np.mean(test_losses_epoch))
        test_accuracies.append(test_accuracy)
        
       


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
