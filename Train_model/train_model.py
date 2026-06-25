import numpy as np 
import tensorflow as tf

print(tf.__version__)
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

train_data = ImageDataGenerator(rescale=1./255)

train_generator = train_data.flow_from_directory(
    "Dataset/Train",
    target_size=(128,128),
    batch_size=32,
    class_mode="binary"
)

test_generator = train_data.flow_from_directory(
    "Dataset/Test",
    target_size=(128,128),
    batch_size=32,
    class_mode="binary"
)

model = Sequential([
    Conv2D(32,(3,3),activation='relu',input_shape=(128,128,3)),
    MaxPooling2D(2,2),

    Conv2D(64,(3,3),activation='relu'),
    MaxPooling2D(2,2),

    Flatten(),

    Dense(128,activation='relu'),

    Dense(1,activation='sigmoid')
])

model.compile(
    optimizer='adam',
    loss='binary_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    train_generator,
    epochs=5,
    validation_data=test_generator
)

model.save("model.h5")

print("Model Saved Successfully")

loss, accuracy = model.evaluate(test_generator)

print("Test Accuracy:", accuracy * 100)

print(train_generator.class_indices)

