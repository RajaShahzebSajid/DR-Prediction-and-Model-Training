from flask import Flask, render_template, request, jsonify
import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adamax
from tensorflow.keras import regularizers
from tensorflow.keras.applications import EfficientNetB3
import numpy as np
from werkzeug.utils import secure_filename
import logging
import pandas as pd
import cv2

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

IMG_SIZE = 224  # Define IMG_SIZE globally
channels = 3
class_count = 4  # Change this to the actual number of classes in your dataset

# Load the pre-trained base model
base_model = EfficientNetB3(include_top=False, weights='imagenet', input_shape=(IMG_SIZE, IMG_SIZE, channels), pooling='max')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_diabetic_retinopathy_model():
    model = Sequential([
        base_model,
        BatchNormalization(axis=-1, momentum=0.99, epsilon=0.001),
        Dense(256, kernel_regularizer=regularizers.l2(0.016), activity_regularizer=regularizers.l1(0.006),
              bias_regularizer=regularizers.l1(0.006), activation='relu'),
        Dropout(rate=0.45, seed=123),
        Dense(class_count, activation='softmax')
    ])
    
    model.compile(Adamax(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
    return model

class_labels_df = pd.read_csv('Diabetic Retinopathy Detection-class_dict.csv')
class_labels = class_labels_df['class_index'].tolist()

# Define the crop_image_from_gray function
def crop_image_from_gray(image):
    if image.ndim == 2:
        mask = image > 7
    else:
        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        mask = gray_img > 7

    if image.ndim == 2:
        img1, img2 = np.where(mask)
        if img1.any() and img2.any():
            image1 = image[np.min(img1):np.max(img1), np.min(img2):np.max(img2)]
        else:
            image1 = image
    else:
        img1, img2 = np.where(mask)
        if img1.any() and img2.any():
            image1 = image[np.min(img1):np.max(img1), np.min(img2):np.max(img2), :]
        else:
            image1 = image

    return image1

# Define the load_ben_color function
def load_ben_color(path, sigmaX=10):
    image = cv2.imread(path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = crop_image_from_gray(image)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    image = cv2.addWeighted(image, 4, cv2.GaussianBlur(image, (0, 0), sigmaX), -4, 128)
    return image

def predict_diabetic_retinopathy(image_path, model):
    img = load_ben_color(image_path)  # Use the load_ben_color function to preprocess the image
    img = img.astype('float32')  # Convert to float32 before normalization
    img = np.expand_dims(img, axis=0)
    img /= 255.  # Normalize the image
    prediction = model.predict(img)
    logging.debug(f"Prediction: {prediction}")
    return prediction

@app.route('/')
def upload_form():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        with open(file_path, "wb") as f:
            f.write(file.read())
        model = load_diabetic_retinopathy_model()
        prediction = predict_diabetic_retinopathy(file_path, model)
        #os.remove(file_path)  # Remove the file after prediction
        predicted_class = class_labels[np.argmax(prediction)]
        result = {
            'prediction': predicted_class,
            'scores': prediction[0].tolist()
        }
        logging.debug(f"Predicted class: {predicted_class}, Confidence scores: {prediction[0]}")
        return jsonify(result)
    else:
        return jsonify({'error': 'Invalid file type'})

if __name__ == '__main__':
    app.run(debug=True)
