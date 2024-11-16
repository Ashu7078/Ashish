import tensorflow as tf
from tensorflow import keras
import streamlit as st
import numpy as np

# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer", 
    "dog", "frog", "horse", "ship", "truck"
]

# Load the pre-trained model
model = keras.models.load_model('cifar101_model.h5')

# Streamlit header
st.header('CIFAR-10 Image Classification Model')

# Define image input parameters
img_height = 32
img_width = 32

# Text input for image file (allow user to upload an image)
image_file = st.file_uploader("Apple.jpg", type=["jpg", "png", "jpeg"])

if image_file is not None:
    # Load and process the uploaded image
    image_load = tf.keras.utils.load_img(image_file, target_size=(img_height, img_width))
    
    # Convert image to numpy array
    img_arr = tf.keras.utils.img_to_array(image_load)
    
    # Normalize image data (model expects images in range [0, 1])
    img_arr = img_arr.astype("float32") / 255.0
    
    # Add batch dimension (model expects 4D input)
    img_bat = np.expand_dims(img_arr, axis=0)
    
    # Predict class
    predictions = model.predict(img_bat)
    
    # Softmax to get class probabilities
    score = tf.nn.softmax(predictions[0])
    
    # Display image and prediction results
    st.image(image_file, caption='Uploaded Image', width=200)
    st.write(f"Predicted Class: {class_names[np.argmax(score)]}")
    st.write(f"Confidence: {np.max(score) * 100:.2f}%")
else:
    st.write("Please upload an image file.")
