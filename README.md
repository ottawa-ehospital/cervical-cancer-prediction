
# Cervical Cancer Prediction Backend

This repository contains the backend for the Cervical Cancer Prediction application, built with Flask. The backend connects to a trained ResNet model to classify cervical cancer images and provide predictions.

---

## Features

- **API Endpoints**:
  - Accepts image uploads via POST requests.
  - Processes images with the trained model to predict cervical cancer classes.
  - Returns classification results in JSON format.

- **Model Integration**:
  - Uses pre-trained machine learning models hosted on Google Drive for inference.
  - Designed for scalability and easy integration with frontend applications.

- **Technologies**:
  - Flask for backend API development.
  - TensorFlow for running the machine learning model.
  - Google Drive (temporarily) for model hosting.

---

## Prerequisites

### Software Requirements
- **Python Version**: Ensure you have Python 3.11.x installed. TensorFlow may encounter compatibility issues with other Python versions.
- **pip**: Python package manager for dependencies.

### Python Libraries
The required Python libraries are listed in `requirements.txt`. Install them with:

```bash
pip install -r requirements.txt
```

---

## Known Compatibility Issues

1. **Python and TensorFlow Compatibility**:
   - This project uses TensorFlow 2.9.1. Ensure that your Python version is compatible.
   - TensorFlow 2.9.1 is officially supported on Python 3.7.x. to 3.11.x Using Python 3.12 or newer may result in installation or runtime errors.
   
2. **Dependency Conflicts**:
   - Conflicts between TensorFlow and other installed libraries can cause errors. Use a virtual environment to avoid issues.
   
3. **Hardware Acceleration**:
   - For GPU acceleration, ensure that your system has compatible CUDA and cuDNN versions installed. Refer to the TensorFlow documentation for detailed GPU setup instructions.

---

## Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/cervical-cancer-prediction.git
cd cervical-cancer-prediction
```

### 2. Create and Activate a Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # For Linux/Mac
venv\Scripts\activate   # For Windows
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Model Files

The application downloads machine learning models from Google Drive. These links are configured in `app.py` under the `MODEL_PATH` variables.

#### Current Implementation:
- Models are accessed via unrestricted Google Drive links.

**Temporary Warning**: Using unrestricted links is not recommended for production due to potential security risks and data quota limits. See the "Recommendations for Production" section below for alternatives.

---

## Usage

### Run the Flask App

```bash
python app.py
```

The Flask server will start, and you can send requests to the API endpoint.

### Example API Usage
- **Endpoint**: `/predict`
- **Method**: POST
- **Body**: Form-data with an image file under the key `file`.

Example using `curl`:

```bash
curl -X POST -F "file=@path_to_image.jpg" http://127.0.0.1:5000/predict
```

The response will include the classification result in JSON format.

---

## Troubleshooting

### 1. TensorFlow Installation Errors
   - Ensure you are using Python 3.9.x for compatibility with TensorFlow 2.9.1.
   - Use the following command if TensorFlow fails to install:

     ```bash
     pip install tensorflow==2.9.1
     ```

### 2. Google Drive Quota Errors
   - If the model download from Google Drive fails due to quota exhaustion, manually download the model file and place it in a `models/` directory in the project root. Update the `MODEL_PATH` in `app.py` to reflect the local path.

### 3. Virtual Environment Issues
   - Activate the virtual environment before running the app to ensure proper dependency isolation.

### 4. GPU Compatibility Issues
   - For systems with GPUs, ensure that you have the correct versions of CUDA and cuDNN installed. Refer to [TensorFlow's GPU support documentation](https://www.tensorflow.org/install/gpu) for details.

---

## Recommendations for Production

1. **Host Models on a Secure Server**:
   - Use cloud storage solutions like AWS S3, Azure Blob Storage, or Google Cloud Storage.
   - Authenticate requests to access the model files.

2. **Use Google Drive API**:
   - Replace unrestricted links with secure programmatic access using the Google Drive API.
   - Create a service account in Google Cloud and use it to fetch files securely.

3. **Avoid Direct Model Links in README**:
   - Avoid publishing model download links in public repositories to prevent quota exhaustion or abuse.

---

## Folder Structure

```plaintext
cervical-cancer-prediction/
├── app.py                  # Flask backend code
├── requirements.txt        # Python dependencies
├── README.md               # Documentation
└── runtime.txt  
└── procfile                
```

---

## Contributing
Contributions are welcome! Please fork this repository, make your changes, and submit a pull request.

---

## License
This project is licensed under the MIT License. See the `LICENSE` file for details.

---

## Closing Notes
This project represents a practical application of machine learning in the healthcare domain. By addressing compatibility challenges and leveraging secure deployment practices, the application is designed to ensure robust performance and ease of use. Together, let's make impactful contributions to predictive healthcare!
