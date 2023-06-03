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
from keras.models import Sequential
from keras.layers import Dense, SimpleRNN,LSTM
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

    X_train, X_test, y_train, y_test = train_test_split(tweets, data.polarity.values, test_size=0.35, random_state=101)
    
    
    # Return the preprocessed data and train/test splits
    return X_train, X_test, y_train, y_test

max_words = 5000
max_len = 200
X_train, X_test, y_train, y_test = preprocess_data("training.1600000.processed.noemoticon.csv",data_limit=100000, max_words = 5000, max_len = 200)
adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-7)

# define the model
model = tf.keras.models.Sequential([
    tf.keras.layers.Embedding(max_words, 128),
    # tf.keras.layers.SimpleRNN(64, activation="tanh",),
    tf.keras.layers.LSTM(64,activation="tanh",dropout=0.2),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(1, activation='sigmoid')
    
])


# compile the model
model.compile(loss='binary_crossentropy', optimizer=adam, metrics=['accuracy'])

# train the model
history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100)

# Save the model and plot the training history
model.save(os.path.join(model_dir, 'lstm.h5'))
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()
plt.savefig(plot_filename)

