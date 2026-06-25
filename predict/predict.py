import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import load_img, img_to_array
# Load the trained model
model = tf.keras.models.load_model("model.h5")

# Enter the image path
img_path = input("Enter image path: ")

# Load and preprocess image
img = load_img(img_path, target_size=(128, 128))
img_array = img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# Predict
prediction = model.predict(img_array)

# Display result
if prediction[0][0] > 0.5:
    print("Prediction: normal")
else:
    print("Prediction: pneumonia")