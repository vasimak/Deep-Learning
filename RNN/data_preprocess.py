import pandas as pd
import numpy as np
import seaborn as sns
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import nltk
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
import string



def preprocess_tweets(data_path, limit=None):
    # Read the CSV file into a DataFrame
    data = pd.read_csv(data_path, encoding='latin', names=['polarity', 'id', 'date', 'query', 'user', 'text'])
    
    # Replace the 'polarity' value of 4 with 1
    data['polarity'] = data['polarity'].replace(4, 1)
    
    # Remove unnecessary columns from the DataFrame
    data.drop(['date', 'query', 'user', 'id'], axis=1, inplace=True)
    
    # Convert the 'text' column to string data type
    data['text'] = data['text'].astype('str')
    
    # Download stopwords from NLTK
    nltk.download('stopwords')
    stopword = set(stopwords.words('english'))
    
    # Download punkt tokenizer from NLTK
    nltk.download('punkt')
    
    # Download WordNetLemmatizer from NLTK
    nltk.download('wordnet')
    
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
    
    # Limit the data if a limit is specified
    if limit is not None:
        data = data.head(limit)
    
    print('Text Preprocessing complete.')
    
    return data


# # # Usage example:
# data_path = "/home/linux/Desktop/Μεταπτυχιακό/Βαθιά Μάθηση και Ανάλυση Πολυμεσικών Δεδομένων/RNN/training.1600000.processed.noemoticon.csv"
# preprocessed_data = preprocess_tweets(data_path,100000)


# data_limit =6000
# def preprocess_data(x, y, num_time_steps):
#     x_new = []
#     y_new = []
#     for i in range(10):
#         indices = np.where(y == i)[0][:data_limit]
#         np.random.shuffle(indices)  # Shuffle the indices before selecting data
#         x_new.append(x[indices])
#         y_new.append(np.full(len(indices), i))
#     x_new = np.concatenate(x_new)
#     y_new = np.concatenate(y_new)
#     x_new = x_new.astype("float32") / 255
#     # reshape input data to have shape (batch_size, timesteps, features)
#     x_new = x_new.reshape(-1, num_time_steps, x_new.shape[2])
#     y_new = np_utils.to_categorical(y_new)
#     indices = np.random.permutation(len(x_new))  # Shuffle the data
#     return x_new[indices], y_new[indices]

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
#num_time_steps = x_train.shape[1]  # number of rows in each image
# X_train, y_train = preprocess_data(x_train, y_train, num_time_steps)
# X_test, y_test = preprocess_data(x_test, y_test, num_time_steps)





# max_words =100000# Number of words to consider as features
# max_len = 500  # Max sequence length (in words)


# # Load the IMDB movie review dataset
# dataset, info = tfds.load('imdb_reviews', with_info=True, as_supervised=True)
# (x_train, y_train), (x_test, y_test) = imdb.load_data(num_words=max_words)

# # Preprocess the data
# X_train = pad_sequences(x_train, maxlen=max_len)
# x_test = pad_sequences(x_test, maxlen=max_len)
# y_train = y_train.reshape(len(y_train), 1)
# y_test = y_test.reshape(len(y_test), 1)
# y_train = to_categorical(y_train)
# y_test = to_categorical(y_test)
# print(X_train.shape,x_test.shape)
# num_time_steps = max_len
# num_features = max_words