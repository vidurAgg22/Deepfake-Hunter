
# 🌟 DeepFake Hunter 🌟

DeepFake Hunter is an intelligent image classification web application designed to detect whether an image is real or a deepfake. The application uses multiple machine learning models trained on a dataset containing real and fake images. Users can upload images and choose which model to use for classification, receiving instant feedback on the authenticity of the image.

## 📋 Table of Contents
- [Features](#features)
- [Technologies Used](#technologies-used)
- [Dataset Structure](#dataset-structure)
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

## 📂 Dataset Structure
The dataset should be organized as follows:
```
Dataset
├── Train
│   ├── Real
│   │   └── (images)
│   └── Fake
│       └── (images)
├── Test
│   ├── Real
│   │   └── (images)
│   └── Fake
│       └── (images)
└── Validation
    ├── Real
    │   └── (images)
    └── Fake
        └── (images)
```

## 📥 Installation
To set up the project locally, follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/deepfake-hunter.git
   cd deepfake-hunter
   ```
2. Create a virtual environment (optional but recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scriptsctivate`
   ```
3. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
4. Ensure the dataset is placed correctly as described above.

## 🚀 Usage
1. Run the Flask application:
   ```bash
   python app.py
   ```
2. Open your web browser and navigate to `http://127.0.0.1:5000`.
3. Upload an image and select a model for classification.
4. View the classification result immediately on the interface.

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a pull request or open an issue to discuss changes.

## 📄 License
This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for more details.
