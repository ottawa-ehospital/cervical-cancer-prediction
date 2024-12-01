
# Cervical Cancer Prediction Backend

This repository contains the backend for the Cervical Cancer Prediction application, built with Flask. The backend connects to a trained ResNet model to classify cervical cancer images and provide predictions.



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


## Prerequisites

### Software Requirements
- Python 3.9 or later
- pip (Python package manager)

### Python Libraries
The required Python libraries are listed in `requirements.txt`. Install them with:

pip install -r requirements.txt


---

## Setup Instructions

### 1. Clone the Repository

git clone https://github.com/yourusername/cervical-cancer-prediction.git
cd cervical-cancer-prediction


### 2. Install Dependencies

pip install -r requirements.txt


### 3. Configure Model Files
The application downloads machine learning models from Google Drive. These links are configured in `app.py` under the `MODEL_PATH` variables.

#### Current Implementation:
- The models are accessed via unrestricted Google Drive links.

**Temporary Warning**: Using unrestricted links is not recommended for production due to potential security risks and data quota limits. See the "Recommendations for Production" section below for alternatives.

---

## Usage

### Run the Flask App

python app.py


The Flask server will start, and you can send requests to the API endpoint.

### Example API Usage
- **Endpoint**: `/predict`
- **Method**: POST
- **Body**: Form-data with an image file under the key `file`.

Example:

curl -X POST -F "file=@path_to_image.jpg" http://127.0.0.1:5000/predict


The response will include the classification result in JSON format.


## Recommendations for Production

For secure and scalable deployment:
1. **Host Models on a Secure Server**:
   - Use cloud storage solutions like AWS S3, Azure Blob Storage, or Google Cloud Storage.
   - Authenticate requests to access the model files.

2. **Use Google Drive API**:
   - Replace unrestricted links with secure programmatic access using the Google Drive API.
   - Create a service account in Google Cloud and use it to fetch files securely.

3. **Avoid Direct Model Links in README**:
   - Avoid publishing model download links in public repositories to prevent quota exhaustion or abuse.


## Known Issues
- **Google Drive Quota**: Frequent downloads of model files may lead to quota exhaustion.
  - Temporary Fix: Cache models locally after the first download.
- **Unrestricted Links**: Using unrestricted Google Drive links is insecure. Transition to authenticated file hosting for production.


## Folder Structure

cervical-cancer-prediction/
├── app.py                  # Flask backend code
├── requirements.txt        # Python dependencies
├── README.md               # Documentation



## Contributing
Contributions are welcome! Please fork this repository, make your changes, and submit a pull request.


## License
This project is licensed under the MIT License. See the `LICENSE` file for details.



### Key Updates in This Version:
1. **Google Drive Model Links**: The README no longer includes explicit links but mentions that the links are configured in `app.py`. This ensures better security and avoids clutter.
2. **Recommendations Section**: Highlights steps for securely hosting and accessing the models.
3. **Folder Structure**: Provides clarity on where to place files and how the project is organized.

