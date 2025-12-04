import streamlit as st
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# Display information about Brain Stroke
def display_intro():
    st.title("Brain Stroke Detection")
    st.write(
        "### What is Brain Stroke?"
        "\nA brain stroke occurs when the blood supply to part of the brain is interrupted, preventing oxygen and nutrients from reaching brain cells. "
        "There are two main types: ischemic (caused by a clot blocking blood flow) and hemorrhagic (caused by bleeding in the brain). "
        "Early detection of stroke is crucial for effective treatment and improved recovery outcomes. Medical imaging techniques such as CT and MRI scans help in diagnosing strokes, "
        "but manual interpretation can be time-consuming and prone to errors. This system aims to assist doctors with automated stroke detection using deep learning."
    )

display_intro()

# Load the trained model
@st.cache_resource
def load_trained_model():
    model = load_model("stroke_detection_model.h5")
    return model

model = load_trained_model()

# Helper function to preprocess the image
def preprocess_image(image):
    """
    Preprocess the uploaded image for prediction.
    Resize to the input size expected by the model (224x224).
    Normalize pixel values to the range [0, 1].
    """
    image = image.resize((224, 224))  # Resize image to (224, 224)
    image = image.convert('RGB')     # Ensure the image has 3 channels
    image_array = np.array(image)
    image_array = image_array / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0)  # Add batch dimension
    return image_array

# Prediction function
def predict_stroke(image):
    """
    Predict whether the image indicates a stroke or not.
    Returns the prediction label (0 = Normal, 1 = Stroke) and confidence score.
    """
    processed_image = preprocess_image(image)
    prediction = model.predict(processed_image)[0][0]  # Sigmoid output
    confidence = prediction  # Direct output is the confidence for "Stroke" class
    class_label = 1 if confidence >= 0.5 else 0
    return class_label, confidence

# Streamlit UI
st.write("Upload a CT scan image to detect if it indicates a stroke.")

# File uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Add a button to trigger prediction
    if st.button("Predict"):
        st.write("Processing...")
        # Make a prediction
        class_label, confidence = predict_stroke(image)
        
        # Display the result with proper confidence interpretation
        if class_label == 0:
            st.success(f"Prediction: **Normal**\nConfidence: {(1 - confidence) * 100:.2f}%")
        else:
            st.error(f"Prediction: **Stroke**\nConfidence: {confidence * 100:.2f}%")

# Display model details at the bottom
st.write("### About the Model")
st.write(
    "This brain stroke detection system uses a Convolutional Neural Network (CNN) based on the **VGG19** architecture. "
    "The model was trained on a dataset of CT scans containing labeled stroke and non-stroke images. "
    "To enhance accuracy, **transfer learning** was applied, utilizing a pre-trained VGG19 model fine-tuned on medical images. "
    "Data augmentation techniques such as image rotation, brightness adjustments, and flipping were used to improve generalization. "
    "This deep learning approach ensures a scalable and reliable solution for automated stroke detection, supporting medical professionals in making quick and accurate diagnoses."
)

