import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt 

# Load MNIST data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

# Plot image data from x_train
plt.imshow(x_train[0], cmap = "gray")
plt.show()

# Assuming you want to use 1% of the labeled data for fine-tuning
# You can adjust this percentage based on your project requirements
percentage_labeled_data = 0.01
num_labeled_samples = int(percentage_labeled_data * len(x_train))

# Create a subset of labeled data for fine-tuning
x_train_labeled = x_train[:num_labeled_samples]
y_train_labeled = y_train[:num_labeled_samples]

# Define the autoencoder model
input_shape = (28, 28, 1)

autoencoder = models.Sequential([
    layers.InputLayer(input_shape=input_shape),
    layers.Conv2D(16, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Conv2D(8, (3, 3), activation='relu', padding='same'),
    layers.MaxPooling2D((2, 2), padding='same'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(32, activation='relu', name='encoded'),
    layers.Dense(64, activation='relu', input_shape=(32,)),  # Adjust input shape
    layers.Reshape((4, 4, 4)),
    layers.Conv2DTranspose(8, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2DTranspose(8, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2DTranspose(16, (3, 3), activation='relu', padding='same'),
    layers.UpSampling2D((2, 2)),
    layers.Conv2DTranspose(1, (3, 3), activation='sigmoid', padding='same')
])

# Compile the autoencoder model
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder on unsupervised data (no labels)
autoencoder.fit(x_train, x_train, epochs=10, batch_size=128, shuffle=True, validation_data=(x_test, x_test))

# Extract the encoder part for feature extraction during fine-tuning
encoder_model = models.Model(inputs=autoencoder.input, outputs=autoencoder.get_layer('encoded').output)

# Define a classifier model for fine-tuning
classifier = models.Sequential([
    encoder_model,  # Use the pretrained encoder
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(10, activation='softmax')  # Assuming a 10-class classification task (MNIST)
])

# Compile the classifier model
classifier.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fine-tune the classifier on a small percentage of labeled data
# You can modify this part based on your specific labeled data
classifier.fit(x_train_labeled, y_train_labeled, epochs=5, batch_size=32, validation_data=(x_test, y_test))
