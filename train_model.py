# train_model.py

import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# I used an older version of tensorflow because new ones wouldn't work on my laptop, says some pipeline issue

# Loading dataset
data = pd.read_csv('dataset/fer2013.csv')

# pixel values are in one string separated by spaces
pixels = data['pixels'].tolist()
x_data = []
for pixel_sequence in pixels:
    x_data.append([int(pixel) for pixel in pixel_sequence.split()])  

x_data = np.array(x_data)
x_data = x_data.reshape((x_data.shape[0], 48, 48, 1))  # 48x48 grayscale images
x_data = x_data / 255.0  # normalize

# Convert emotion labels to one-hot encoding
y_data = to_categorical(data['emotion'], num_classes=7)

# Split the dataset into training and testing
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size=0.2, random_state=42)


model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # added dropout so it wouldn't overfit
model.add(Dense(7, activation='softmax'))  # 7 emotions in the dataset

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model (my laptop was slow so I used 10 epochs only)
model.fit(x_train, y_train, epochs=10, batch_size=64, validation_data=(x_test, y_test))

# Save the trained model to a file
model.save('model/emotion_model.h5')  

print("Model trained and saved")
