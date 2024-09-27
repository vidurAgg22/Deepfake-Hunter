# 🌟 DeepFake Hunter 🌟

DeepFake Hunter is an intelligent image classification web application designed to detect whether an image is real or a deepfake. The application uses multiple machine learning models trained on a dataset containing real and fake images. Users can upload images and choose which model to use for classification, receiving instant feedback on the authenticity of the image.

## 📋 Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## ✨ Features
- 🔍 Upload images for classification as either **Real** or **Fake**.
- ⚙️ Select from multiple trained models for prediction.
- 💻 Easy-to-use web interface built with **Flask** and **Bootstrap**.
- ⏱️ Real-time feedback on image classification.

## 🛠️ Technologies Used
- **Flask**: A lightweight WSGI web application framework for Python.
- **scikit-learn**: A machine learning library for Python, providing tools for model training and evaluation.
- **NumPy**: A library for numerical operations in Python.
- **PIL (Pillow)**: A library for image processing in Python.
- **joblib**: A library for saving and loading machine learning models.
- **Bootstrap**: A front-end framework for developing responsive web applications.

## 📂 Project Structure
Due to GitHub size constraints, the dataset and trained models are not included in the repository. Here's the complete project structure, including where to place these files:

```
DeepFakeHunter/
├── app.py                 # Main Flask application
├── train_models.py        # Script to train and save models
├── requirements.txt       # Python dependencies
├── .gitignore             # Specifies files and folders to ignore in Git
├── README.md              # This file
├── static/
│   ├── styles.css         # CSS file for styling
│   └── uploads/           # Folder for storing uploaded files (created automatically)
├── templates/
│   └── index.html         # HTML file for the front end
├── models/                # Folder for storing trained models (created by train_models.py)
│   ├── model1.joblib
│   ├── model2.joblib
│   └── ...
└── dataset/               # Folder for your dataset (not included in repo)
    ├── Train/
    │   ├── Real/
    │   │   └── (images)
    │   └── Fake/
    │       └── (images)
    ├── Test/
    │   ├── Real/
    │   │   └── (images)
    │   └── Fake/
    │       └── (images)
    └── Validation/
        ├── Real/
        │   └── (images)
        └── Fake/
            └── (images)
```

**Note**: The `dataset/` folder is not included in the repository. You'll need to download the dataset(https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) and add it manually after cloning the repository. The `models/` folder will be created and populated when you run `train_models.py`.

## 📥 Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/vidurAgg22/deepfake-hunter.git
   cd deepfake-hunter
   ```

2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the dataset from [Kaggle](https://www.kaggle.com/datasets/manjilkarki/deepfake-and-real-images) and place it in the `dataset/` folder, following the structure shown above.

5. Train the models by running:
   ```bash
   python train_models.py
   ```
   This will create the `models/` folder and populate it with trained models.

## 🚀 Usage
1. Ensure you have completed all the installation steps, including adding the dataset and training the models.

2. Run the Flask application:
   ```bash
   python app.py
   ```

3. Open your web browser and navigate to `http://127.0.0.1:5000`.

4. Upload an image and select a model for classification.

5. View the classification result immediately on the interface.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss changes or improvements.
