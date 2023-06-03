import numpy as np
from Dropout_layer import *
from Softmax import *
from Loss_function import *
from Dense_layer import *
from tqdm import tqdm
import matplotlib.pyplot as plt

from keras.datasets import mnist
from keras.utils import np_utils
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import imdb
from keras.utils import np_utils
from tensorflow.keras.preprocessing.sequence import pad_sequences
import pandas as pd
from keras.utils import to_categorical
from keras.preprocessing.text import Tokenizer
from keras import regularizers
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string
import os

from sklearn.model_selection import train_test_split
# Define the path to the directory where the model will be saved
model_dir = 'saved_models'
plot_filename = 'lstm_png'
os.makedirs(model_dir, exist_ok=True)
def preprocess_data(filename, data_limit=None,max_words=0,max_len=0):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(filename, encoding='latin', names=['polarity', 'id', 'date', 'query', 'user', 'text'])
    
    # Shuffle the data
    data = data.sample(frac=1)
    
    # Limit the data if specified
    if data_limit is not None:
        data = data[:data_limit]
    
    # Replace the 'polarity' value of 4 with 1
    data['polarity'] = data['polarity'].replace(4, 1)
    
    # Remove unnecessary columns from the DataFrame
    data.drop(['date', 'query', 'user', 'id'], axis=1, inplace=True)
    
    # Convert the 'text' column to string data type
    data['text'] = data['text'].astype('str')
    
    # Download stopwords from NLTK
    # nltk.download('stopwords')
    stopword = set(stopwords.words('english'))
    
    # Download punkt tokenizer from NLTK
    # nltk.download('punkt')
    
    # Download WordNetLemmatizer from NLTK
    # nltk.download('wordnet')
    
    # Define regex patterns for URL and username removal
    urlPattern = r"((http://)[^ ]*|(https://)[^ ]*|( www\.)[^ ]*)"
    userPattern = '@[^\s]+'
    
    def process_tweets(tweet):
        # Convert the tweet to lowercase
        tweet = tweet.lower()
        # Remove the first character (assuming it's a special character)
        tweet = tweet[1:]
        # Remove URLs using regex pattern
        tweet = re.sub(urlPattern, '', tweet)
        # Remove usernames using regex pattern
        tweet = re.sub(userPattern, '', tweet)
        # Remove punctuation marks
        tweet = tweet.translate(str.maketrans("", "", string.punctuation))
        # Tokenize the tweet into words
        tokens = word_tokenize(tweet)
        # Remove stopwords from the tokenized words
        final_tokens = [w for w in tokens if w not in stopword]
        # Lemmatize the words
        wordLemm = WordNetLemmatizer()
        finalwords = []
        for w in final_tokens:
            if len(w) > 1:
                word = wordLemm.lemmatize(w)
                finalwords.append(word)
        # Join the final words back into a processed tweet
        return ' '.join(finalwords)
    
    # Apply the tweet processing function to the 'text' column and store the processed tweets in a new column 'processed_tweets'
    data['processed_tweets'] = data['text'].apply(lambda x: process_tweets(x))
    
    print('Text Preprocessing complete.')
    
 
    
    tokenizer = Tokenizer(num_words=max_words)
    tokenizer.fit_on_texts(data.processed_tweets)
    sequences = tokenizer.texts_to_sequences(data.processed_tweets)
    tweets = pad_sequences(sequences, maxlen=max_len)
    X_train, X_test, y_train, y_test = train_test_split(tweets, data.polarity.values,data_limit=100, test_size=0.35, random_state=101)
    
    
    # Return the preprocessed data and train/test splits
    return X_train, X_test, y_train, y_test

max_words = 5000
max_len = 200
X_train, X_test, y_train, y_test = preprocess_data("RNN/training.1600000.processed.noemoticon.csv",max_words = 5000, max_len = 200)

class Loss:
    def remember_trainable_layers(self,trainable_layers):
        self.trainable_layers = trainable_layers
        
    def calculate(self,output,y):
        sample_losses= self.forward(output,y)

        data_loss = np.mean(sample_losses)


        return data_loss

class Loss_CategoricalCrossentropy(Loss):
    def forward(self,y_pred,y_true):
        samples = len(y_pred)
        y_pred_clipped = np.clip(y_pred, 10e-3, 1.0 - 10e-3)
      
        # print(y_true.shape)
        if len(y_true.shape) == 1:
            correct_confidences = y_pred_clipped [range(samples), y_true]
        
        elif len(y_true.shape) == 2:
            correct_confidences = np.sum(y_pred_clipped*y_true,axis=1)
        negative_log_likelihood = -np.log(correct_confidences)
        return negative_log_likelihood
    
    def backwards(self,d_output,y_true):
        samples =len(d_output)
        labels = len(d_output[0])
        # print(d_output)
        # print(y_true)
        if len(y_true.shape) == 1:
            print('yehaw')
            y_true = np.eye(labels)[y_true]

        self.d_inputs = -y_true/d_output

        self.d_inputs = self.d_inputs /samples

class Activation_Softmax:

    def __init__(self):
        self.predictions = []
    def forward(self,inputs):
        self.inputs = inputs
       
        exp_values = np.exp(inputs-np.max(inputs, axis=1, keepdims=True))
        probabilities = exp_values / np.sum(exp_values, axis=1, keepdims=True)
        self.output = probabilities
        self.predictions.append(probabilities)
     
        

    def backwards(self,d_output):
        self.d_inputs = np.empty_like(d_output)
        
        for index, (single_output,single_d_output) in enumerate(zip(self.output,d_output)):
            single_output = single_output.reshape(-1,1)

            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output,single_output.T)

            self.d_inputs[index] = np.dot(jacobian_matrix,single_d_output)
  

    def predictions(self,output):
        
        return np.argmax(output,axis=1)


from numpy.random import randn

class RNN:
  # A many-to-one Vanilla Recurrent Neural Network.

  def __init__(self, input_size, output_size, hidden_size=64):
    # Weights
    self.Whh = randn(hidden_size, hidden_size) / 1000
    self.Wxh = randn(hidden_size, input_size) / 1000
    self.Why = randn(output_size, hidden_size) / 1000

    # Biases
    self.bh = np.zeros((hidden_size, 28))
    self.by = np.zeros((output_size, 28))

  def forward(self, inputs):
    '''
    Perform a forward pass of the RNN using the given inputs.
    Returns the final output and hidden state.
    - inputs is an array of one hot vectors with shape (input_size, 1).
    '''
    h = np.zeros((self.Whh.shape[0], 1))

    self.last_inputs = inputs
    self.last_hs = { 0: h }

    # Perform each step of the RNN
    for i, x in enumerate(inputs):
      h = np.tanh(self.Wxh @ x + self.Whh @ h + self.bh)
      self.last_hs[i + 1] = h

    # Compute the output
    y = self.Why @ h + self.by
    self.outputs =y
    print(h.shape,y.shape)

    return y ,h

  def backprop(self, d_y, learn_rate=2e-2):
    '''
    Perform a backward pass of the RNN.
    - d_y (dL/dy) has shape (output_size, 1).
    - learn_rate is a float.
    '''
    n = len(self.last_inputs)

    # Calculate dL/dWhy and dL/dby.
    d_Why = d_y @ self.last_hs[n].T
    d_by = d_y

    # Initialize dL/dWhh, dL/dWxh, and dL/dbh to zero.
    d_Whh = np.zeros(self.Whh.shape)
    d_Wxh = np.zeros(self.Wxh.shape)
    d_bh = np.zeros(self.bh.shape)
    print(d_bh.shape)

    # Calculate dL/dh for the last h.
    # dL/dh = dL/dy * dy/dh
    d_h = self.Why.T @ d_y

    # Backpropagate through time.
    for t in reversed(range(n)):
      # An intermediate value: dL/dh * (1 - h^2)
      temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)
    
      # dL/db = dL/dh * (1 - h^2)
      d_bh += temp

      # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
      d_Whh += np.dot(temp.T ,self.last_hs[t])

      # dL/dWxh = dL/dh * (1 - h^2) * x
      d_Wxh += temp @ self.last_inputs[t].T

      # Next dL/dh = dL/dh * (1 - h^2) * Whh
      d_h = self.Whh @ temp

    # Clip to prevent exploding gradients.
    for d in [d_Wxh, d_Whh, d_Why, d_bh, d_by]:
      np.clip(d, -1, 1, out=d)

    # Update weights and biases using gradient descent.
    self.Whh -= learn_rate * d_Whh
    self.Wxh -= learn_rate * d_Wxh
    self.Why -= learn_rate * d_Why
    self.bh -= learn_rate * d_bh
    self.by -= learn_rate * d_by









num_features = len(X_train[0])

# define RNN layer
rnn = RNN(input_size=num_features, hidden_size=64, output_size=100)
dense = Layer_Dense(28, 128)
activation = Activation_ReLU()
dense2 = Layer_Dense(128,10)
activation2 = Activation_ReLU()
softmax = Activation_Softmax()
loss_function = Loss_CategoricalCrossentropy()

# Train the model for 10 epochs
for epoch in range(1):
    # Forward propagate
    Y_pred,_= rnn.forward(X_train)
    dense.forward(Y_pred)
    activation.forward((dense.output))
    dense2.forward(activation.output)
    activation2.forward((dense2.output))

    softmax.forward(activation2.output)
    
    # Calculate loss
    loss_function.forward(softmax.output, y_train)

    # Calculate gradients
    loss_function.backwards(softmax.output, y_train)
    softmax.backwards(loss_function.d_inputs)
    activation2.backwards(softmax.d_inputs)
    dense2.backwards(activation2.d_inputs)
    activation.backwards(dense2.d_inputs)
    dense.backwards(activation.d_inputs)
    rnn.backprop(dense.d_inputs)
    
    # Print train accuracy every 10 epochs
    if epoch % 10 == 0:
        Y_pred = rnn.forward(X_train)
        softmax.forward(Y_pred)
        predictions = np.argmax(softmax.output, axis=1)
   
        if len(y_train.shape) == 2:
            train_labels=np.argmax(y_train,axis=1)
        accuracy = np.mean(predictions ==train_labels)
        print(f"Train accuracy at epoch {epoch}: {accuracy}")



# class RNN:
#     def __init__(self, input_size, hidden_size, output_size, timesteps=1, batch_size=1,learning_rate=10e-5):
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.output_size = output_size
#         self.timesteps = timesteps
#         self.batch_size = batch_size
#         self.learning_rate = learning_rate
#         # initialize weights and biases
#         self.Wxh = np.random.randn(hidden_size, input_size) * 0.01
#         self.Whh = np.random.randn(hidden_size, hidden_size) * 0.01
#         self.Why = np.random.randn(output_size, hidden_size) * 0.01
#         self.bh = np.zeros((1, hidden_size))
#         self.by = np.zeros((1, output_size))
        
#         # initialize hidden state and cell state
#         self.h = np.zeros((batch_size, timesteps, hidden_size))
#         self.activations = np.zeros((batch_size, timesteps, hidden_size))
#         self.X = None
        
#     def forward(self, X):
#         self.X = X
#         for t in range(self.timesteps):
#             if t == 0:
#                 self.h[:, t, :] = np.tanh(
#                     np.dot(X[:, t, :], self.Wxh.T) + self.bh
#                 )
#             else:
#                 self.h[:, t, :] = np.tanh(
#                     np.dot(X[:, t, :], self.Wxh.T) + np.dot(self.h[:, t-1, :], self.Whh) + self.bh
#                 )
#             self.activations[:, t, :] = self.h[:, t, :]
        
#         out = np.dot(self.h[:, -1, :], self.Why.T) + self.by
#         print(out)
#         return out
    
#     def backward(self, dY):
#         dWhy = np.dot(dY.T, self.h[:, -1, :])
#         dby = np.sum(dY, axis=0, keepdims=True)
#         d_hidden_prev = np.zeros((self.batch_size, self.timesteps, self.hidden_size))

#         dWxh, dWhh, dbh = np.zeros_like(self.Wxh), np.zeros_like(self.Whh), np.zeros_like(self.bh)
     

       
#                 # Backpropagate through time
#         for t in reversed(range(self.timesteps)):
#             # Add gradients from output layer

#             temp = ((1 - self.last_hs[t + 1] ** 2) * d_h)

#             # dL/db = dL/dh * (1 - h^2)
#             dbh += temp
#             # dL/dWhh = dL/dh * (1 - h^2) * h_{t-1}
#             dWhh += temp @ self.last_hs[t].T
#             # dL/dWxh = dL/dh * (1 - h^2) * x
#             dWxh += temp @ self.last_inputs[t].T
#             # Next dL/dh = dL/dh * (1 - h^2) * Whh
#             dh = self.Whh @ temp
#             # d_h = np.dot(dY[t], self.Why) + d_hidden_prev[:, t, :]

#             # # Backpropagate through activation function
#             # d_a = d_h * (1 - self.h[:, t, :] ** 2)
           
#             # print(self.X[:, t, :].reshape(self.batch_size, -1).shape,d_a.shape)
#             # n=self.X[:, t, :].reshape(self.batch_size, -1)
#             # # Calculate gradients for parameters
#             # dWxh += np.dot(n.T, d_a)


#             dWhh += np.dot(self.h[:, t-1, :], d_a)
#             dbh += np.sum(d_a, axis=0, keepdims=True)

#             # Calculate gradients for previous hidden state
#             d_hidden_prev[:, t-1, :] = np.dot(d_a, self.Whh.T)

#         # Clip gradients to avoid exploding gradients
#         for dparam in [dWxh, dWhh, dWhy, dbh, dby]:
#             np.clip(dparam, -5, 5, out=dparam)

#         # Update parameters
#         self.Wxh -= self.learning_rate * dWxh
#         self.Whh -= self.learning_rate * dWhh
#         self.Why -= self.learning_rate * dWhy
#         self.bh -= self.learning_rate * dbh
#         self.by -= self.learning_rate * dby
#         # print(self.Wxh, self.Whh, self.Why)