from flask import Flask, request, jsonify, url_for
import os
import numpy as np
import cv2
import gdown
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from flask_cors import CORS

# Flask app configuration
app = Flask(__name__, static_url_path='/static')
CORS(app)

# Configure upload folders
UPLOAD_FOLDER = os.path.join(app.root_path, 'static', 'uploads')
OUTPUT_FOLDER = os.path.join(app.root_path, 'static', 'output')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Initialize a dictionary to hold models
models = {}

# Define the model download URLs (replace with your actual links)
MODEL_URLS = {
    'DenseNet169': 'https://drive.google.com/uc?export=download&id=1cuMNlSPYB65f9qHtXFgSMbpdBI6csn6C',
    #'ResNet101': 'https://drive.google.com/uc?export=download&id=1Se_4jG8wS5FMGTrC5DXF0K2pT0P4hJrT',
    #'XceptionNet': 'https://drive.google.com/uc?export=download&id=1fwi4WWgZvZBkXDLRSFhYL5QHgcT_aYFZ',
}

def download_model_from_google_drive(url, output):
    """Downloads and loads a model from Google Drive."""
    try:
        if not os.path.exists(output):
            print(f"Downloading model from {url}...")
            gdown.download(url, output, quiet=False, fuzzy=True)
        print(f"Loading model from {output}...")
        model = load_model(output)
        return model
    except Exception as e:
        raise Exception(f"Failed to download or load model from {url}: {str(e)}")

def load_models():
    """Load all models from the defined URLs."""
    for model_name, model_url in MODEL_URLS.items():
        output = f"{model_name}.h5"
        try:
            if model_name not in models:
                print(f"Processing model: {model_name}")
                models[model_name] = download_model_from_google_drive(model_url, output)
        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")

# Load models at application start
load_models()

def preprocess_image(image_path):
    """Preprocess the image for model prediction."""
    img = image.load_img(image_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    original_img = cv2.cvtColor(img_array.astype(np.uint8), cv2.COLOR_RGB2BGR)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array, original_img

def apply_filters(original_img):
    """Apply image filters to improve model accuracy."""
    median_filtered = cv2.medianBlur(original_img, 5)
    nlm_filtered = cv2.fastNlMeansDenoisingColored(median_filtered, None, 10, 10, 7, 21)
    return nlm_filtered

def predict_cancer_type(model, img_array):
    """Predict cancer type from the processed image."""
    prediction = model.predict(img_array)
    class_index = np.argmax(prediction)
    return class_index

def determine_cancerous(class_index):
    """Map the class index to class name and determine if it is cancerous."""
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
    """Endpoint for image preprocessing."""
    file = request.files.get("file")
    if not file:
        return jsonify({"error": "No image file uploaded."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        img_array, original_img = preprocess_image(filepath)
        filtered_img = apply_filters(original_img)

        # Save the preprocessed image
        filtered_image_filename = f'preprocessed_{filename}'
        filtered_image_path = os.path.join(OUTPUT_FOLDER, filtered_image_filename)
        cv2.imwrite(filtered_image_path, filtered_img)

        # Generate URLs for images
        original_image_url = url_for('static', filename=f'uploads/{filename}', _external=True)
        filtered_image_url = url_for('static', filename=f'output/{filtered_image_filename}', _external=True)

        return jsonify({
            "original_image": original_image_url,
            "preprocessed_image": filtered_image_url,
            "message": "Images processed successfully."
        })
    except Exception as e:
        return jsonify({"error": f"Error during preprocessing: {str(e)}"}), 500

@app.route("/predict", methods=["POST"])
def predict():
    """Endpoint for making predictions."""
    model_name = request.form.get("model")
    file = request.files.get("file")
    if model_name not in models:
        return jsonify({"error": f"Model '{model_name}' not found."}), 400
    if not file:
        return jsonify({"error": "No image file uploaded."}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        img_array, original_img = preprocess_image(filepath)
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
    port = int(os.environ.get("PORT", 5000))  
    app.run(host="0.0.0.0", port=port, debug=False)
