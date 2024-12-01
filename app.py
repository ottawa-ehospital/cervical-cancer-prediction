from flask import Flask, request, jsonify, send_from_directory
import os
import numpy as np
import cv2
import gdown
from io import BytesIO
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Flask app configuration
app = Flask(__name__, static_url_path='/static')
CORS(app)

# Configure upload folder
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/output')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize a dictionary to hold models
models = {}

# Define the model download URLs (replace with your actual links)
MODEL_URLS = {
    'DenseNet169': r'https://drive.google.com/uc?export=download&id=Y1cuMNlSPYB65f9qHtXFgSMbpdBI6csn6C&confirm=t',
    'ResNet101': r'https://drive.google.com/uc?export=download&id=1Se_4jG8wS5FMGTrC5DXF0K2pT0P4hJrT&confirm=t',
    'XceptionNet': r'https://drive.google.com/uc?export=download&id=1fwi4WWgZvZBkXDLRSFhYL5QHgcT_aYFZ&confirm=t',
}

def download_model_from_google_drive(url, output):
    try:
        # Download model using gdown
        gdown.download(url, output, quiet=False)

        # Load the model after downloading
        model = load_model(output)
        return model
    except Exception as e:
        raise Exception(f"Failed to download model from {url}: {str(e)}")

def load_models():
    for model_name, model_url in MODEL_URLS.items():
        output = f"{model_name}.h5"
        print(f"Downloading and loading model: {model_name}")
        models[model_name] = download_model_from_google_drive(model_url, output)

# Load models at application start
load_models()

def preprocess_image(image_path):
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    original_img = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, original_img

def apply_filters(original_img):
    median_filtered = cv2.medianBlur(original_img, 5)
    gray_img = cv2.cvtColor(median_filtered, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    clahe_img = clahe.apply(gray_img)
    clahe_img = cv2.cvtColor(clahe_img, cv2.COLOR_GRAY2BGR)
    nlm_filtered = cv2.fastNlMeansDenoisingColored(median_filtered, None, 10, 10, 7, 21)
    return nlm_filtered

def predict_cancer_type(model, img_array):
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    return class_index

def determine_cancerous(class_index):
    class_names = [
        'carcinoma_in_situ', 'light_dysplastic', 'moderate_dysplastic', 
        'normal_columnar', 'normal_intermediate', 'normal_superficiel', 'severe_dysplastic'
    ]
    cancerous_classes = {
        'carcinoma_in_situ': True, 'light_dysplastic': True, 'moderate_dysplastic': True,
        'normal_columnar': False, 'normal_intermediate': False, 'normal_superficiel': False,
        'severe_dysplastic': True,
    }
    class_name = class_names[class_index]
    is_cancerous = cancerous_classes.get(class_name, None)
    return class_name, is_cancerous

@app.route("/preprocess", methods=["POST"])
def preprocess():
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No image file uploaded."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        img_array, original_img = preprocess_image(filepath)
        filtered_img = apply_filters(original_img)

        filtered_image_path = os.path.join(app.config['UPLOAD_FOLDER'], 'filtered_' + filename)
        cv2.imwrite(filtered_image_path, filtered_img)

        return jsonify({
            "original_image": filepath,
            "preprocessed_image": filtered_image_path
        })

    except Exception as e:
        return jsonify({"error": f"Error during preprocessing: {str(e)}"}), 500

@app.route("/predict", methods=["POST"])
def predict():
    model_name = request.form.get("model")
    file = request.files.get("file")
    age = request.form.get("age")
    smoking_status = request.form.get("smoking_status")

    if model_name not in models:
        return jsonify({"error": f"Model '{model_name}' not found."}), 400

    if not file:
        return jsonify({"error": "No image file uploaded."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        img_array, original_img = preprocess_image(filepath)
        filtered_img = apply_filters(original_img)
        class_index = predict_cancer_type(models[model_name], img_array)
        class_name, is_cancerous = determine_cancerous(class_index)

        risk = "High" if is_cancerous else "Low"
        recommendation = (
            "Schedule a screening and consult a healthcare provider."
            if is_cancerous else
            "Maintain regular check-ups and a healthy lifestyle."
        )

        return jsonify({
            "class_name": class_name,
            "is_cancerous": is_cancerous,
            "risk": risk,
            "recommendation": recommendation
        })

    except Exception as e:
        return jsonify({"error": f"Error during prediction: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True)
