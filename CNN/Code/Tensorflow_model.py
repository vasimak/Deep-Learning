import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import SGD
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.datasets import mnist
import matplotlib.pyplot as plt
# Set data limits and filenames
import os

# Define the path to the directory where the model will be saved
model_dir = 'saved_models'
os.makedirs(model_dir, exist_ok=True)

# Save the model inside the directory

data_limit = 4000
plot_filename = "tensorflow_lr_moment_decay.png"
# model_output = "model.txt"

# Define a function to preprocess the data
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
    x_new = x_new.reshape(len(x_new), 28, 28, 1)
    x_new = x_new.astype("float32") / 255
    y_new = to_categorical(y_new, num_classes=10)
    indices = np.random.permutation(len(x_new))  # Shuffle the data
    return x_new[indices], y_new[indices]

# Load the MNIST dataset and preprocess it
(x_train, y_train), (x_test, y_test) = mnist.load_data()
X_train, y_train = preprocess_data(x_train, y_train)
X_test, y_test = preprocess_data(x_test, y_test)

# Define the neural network architecture
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dropout(0.2),
    # tf.keras.layers.Dense(128, activation='relu'),
    
    tf.keras.layers.Dense(10, activation='softmax'),
    tf.keras.layers.Dropout(0.2)
])

opt = tf.keras.optimizers.legacy.SGD(learning_rate=10e-5)

# opt = SGD(learning_rate=10e-5)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
# Train the model
history = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=100,batch_size=64)

# Save the model and plot the training history
model.save(os.path.join(model_dir, 'tensor.h5'))
plt.plot(history.history['accuracy'], label='training accuracy')
plt.plot(history.history['val_accuracy'], label='validation accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.savefig(plot_filename)
