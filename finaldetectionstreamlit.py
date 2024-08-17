import streamlit as st
import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from tensorflow.keras.models import load_model, save_model, Model
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.utils import to_categorical
from PIL import Image
import os

# Function to compute LBP features
def compute_lbp(image, radius=1, n_points=8, method='uniform'):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = local_binary_pattern(gray_image, n_points, radius, method)
    lbp = np.uint8((lbp / np.max(lbp)) * 255)
    return lbp

# Function to preprocess the image
def preprocess_image(image):
    image = cv2.resize(image, (224, 224))
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    return image

# Function to load the latest model
def load_latest_model():
    model_path = 'C:/Users/SHAHZEIB/Downloads/final/latest_model.h5'  # Path to the latest model
    if os.path.exists(model_path):
        model = load_model(model_path)
    else:
        # If model does not exist, load the default pretrained model
        model_path = 'C:/Users/SHAHZEIB/Downloads/final/i-Diabetic Retinopathy Detection-93.85.h5'
        model = load_model(model_path)
    return model

# Function to save the model
def save_trained_model(model):
    model_path = 'C:/Users/SHAHZEIB/Downloads/final/latest_model.h5'
    save_model(model, model_path)

# Initialize session state variables
if 'page' not in st.session_state:
    st.session_state.page = 'home'
if 'severity_probabilities' not in st.session_state:
    st.session_state.severity_probabilities = None
if 'roi_image_resized' not in st.session_state:
    st.session_state.roi_image_resized = None
if 'lbp_features_resized' not in st.session_state:
    st.session_state.lbp_features_resized = None

# Define pages
def home_page():
    st.title("🩺 Diabetic Retinopathy Detection")

    # Welcome message with animation
    st.markdown("""
        <div style="text-align: center;">
            <h3>Welcome to the Diabetic Retinopathy Detection App.</h3>
            <p>Upload a retinal image to detect the severity of diabetic retinopathy.</p>
        </div>
        """, unsafe_allow_html=True)

    # Upload image
    uploaded_file = st.file_uploader("Choose a retinal image...", type=["jpg", "jpeg", "png"])

    # Dropdown for selecting labels
    selected_label = st.selectbox("Select Label for Training", ['NODR', 'MODERATE', 'SEVERE'])

    # Mapping labels to numerical values
    label_map = {'NODR': 0, 'MODERATE': 1, 'SEVERE': 2}

    # Option to train the model
    if st.button('Train Model'):
        st.subheader('Training the Model')

        if uploaded_file is not None:
            try:
                # Load the image
                image = Image.open(uploaded_file)
                image = np.array(image)

                # Display the original size image with animation
                st.image(image, caption='Uploaded Image', use_column_width=False)

                # Convert image to grayscale
                gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

                # Perform color thresholding to focus on retinal area
                ret, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

                # Find contours in the thresholded image
                contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Get the bounding box of the largest contour (assuming it's the retinal area)
                if contours:
                    largest_contour = max(contours, key=cv2.contourArea)
                    x, y, w, h = cv2.boundingRect(largest_contour)

                    # Define region of interest (ROI) based on the bounding box
                    roi_image = image[y:y+h, x:x+w]

                    # Resize ROI image to a fixed size for display and analysis
                    roi_image_resized = cv2.resize(roi_image, (300, 300))

                    # Display ROI with animation
                    st.subheader("Region of Interest (ROI)")
                    st.image(roi_image_resized, caption='ROI Image', use_column_width=False)

                    # Compute LBP features for ROI
                    lbp_features = compute_lbp(roi_image)

                    # Resize LBP features image to the same size
                    lbp_features_resized = cv2.resize(lbp_features, (300, 300))

                    # Display LBP features with animation
                    st.subheader("Extracted LBP Features")
                    st.image(lbp_features_resized, caption='LBP Features', use_column_width=False)

                    # Preprocess the ROI image for model input
                    processed_roi = preprocess_image(roi_image)

                    # Load the latest model
                    model = load_latest_model()

                    # Prepare the label
                    label = label_map[selected_label]
                    label = to_categorical(label, num_classes=3)
                    label = np.expand_dims(label, axis=0)

                    # Define callbacks to save model checkpoints
                    checkpoint_path = 'C:/Users/SHAHZEIB/Downloads/final/model_checkpoint.h5'  # Adjust path if needed
                    checkpoint_callback = ModelCheckpoint(filepath=checkpoint_path, save_weights_only=False, save_best_only=True)

                    # Train the model with the single image and label
                    epochs = 5
                    st.progress(0)
                    for epoch in range(epochs):
                        st.write(f"Epoch {epoch + 1}/{epochs}: Training in progress...")
                        model.fit(processed_roi, label, epochs=1, verbose=0, callbacks=[checkpoint_callback])
                        st.progress((epoch + 1) / epochs)

                    # Save the trained model
                    save_trained_model(model)

                    st.success('Training completed successfully.')

                else:
                    st.warning("No contours detected. Unable to determine ROI.")

            except Exception as e:
                st.error(f"An error occurred: {e}")

        else:
            st.warning("Please upload an image to train the model.")

    # Continue with image analysis functionality
    if uploaded_file is not None:
        try:
            # Load the image
            image = Image.open(uploaded_file)
            image = np.array(image)

            # Display the original size image with animation
            st.image(image, caption='Uploaded Image', use_column_width=False)

            # Convert image to grayscale
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

            # Perform color thresholding to focus on retinal area
            ret, thresh = cv2.threshold(gray_image, 50, 255, cv2.THRESH_BINARY)

            # Find contours in the thresholded image
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            # Get the bounding box of the largest contour (assuming it's the retinal area)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(largest_contour)

                # Define region of interest (ROI) based on the bounding box
                roi_image = image[y:y+h, x:x+w]

                # Resize ROI image to a fixed size for display and analysis
                roi_image_resized = cv2.resize(roi_image, (300, 300))

                # Compute LBP features for ROI
                lbp_features = compute_lbp(roi_image)

                # Resize LBP features image to the same size
                lbp_features_resized = cv2.resize(lbp_features, (300, 300))

                # Preprocess the ROI image for model input
                processed_roi = preprocess_image(roi_image)

                # Load the latest model
                model = load_latest_model()

                # Predict severity probabilities for the input image using the latest model
                if st.button('Predict'):
                    with st.spinner('Analyzing the image...'):
                        severity_probabilities = model.predict(processed_roi)
                    
                    # Store the necessary data in session state
                    st.session_state.severity_probabilities = severity_probabilities
                    st.session_state.roi_image_resized = roi_image_resized
                    st.session_state.lbp_features_resized = lbp_features_resized
                    st.session_state.page = 'results'
                    st.experimental_rerun()

        except Exception as e:
            st.error(f"An error occurred: {e}")

    else:
        st.info("Please upload an image to analyze.")

def results_page():
    st.title("Prediction Results")

    # Define descriptive labels
    class_labels = ["No Dr", "Moderate", "Severe"]

    # Retrieve the severity probabilities and images from session state
    severity_probabilities = st.session_state.severity_probabilities
    roi_image_resized = st.session_state.roi_image_resized
    lbp_features_resized = st.session_state.lbp_features_resized

    # Display ROI Image
    st.subheader("Region of Interest (ROI)")
    st.image(roi_image_resized, caption='ROI Image', use_column_width=False)

    # Display LBP Features
    st.subheader("Extracted LBP Features")
    st.image(lbp_features_resized, caption='LBP Features', use_column_width=False)

    # Display predicted severity probability with the highest percentage
    st.subheader("Predicted Severity Probability")
    if severity_probabilities is not None:
        severity_probabilities_percentage = severity_probabilities[0] * 100
        max_index = np.argmax(severity_probabilities_percentage)
        max_label = class_labels[max_index]
        max_probability = severity_probabilities_percentage[max_index]
        st.markdown(f"- **{max_label}**")
    else:
        st.warning("No prediction results available.")

    if st.button('Go Back to Home'):
        st.session_state.page = 'home'
        st.experimental_rerun()

# Page navigation
if st.session_state.page == 'home':
    home_page()
elif st.session_state.page == 'results':
    results_page()
