from flask import Flask, render_template, request, redirect, url_for
import os
import numpy as np
from PIL import Image
from joblib import load

app = Flask(__name__)

# Paths for the models
model_paths = {
    'KNN': os.path.join('model', 'knn_model.joblib'),
    'SVM': os.path.join('model', 'svm_model.joblib'),
    'Random Forest': os.path.join('model', 'rf_model.joblib')
}

# Load models
models = {name: load(path) for name, path in model_paths.items()}

IMAGE_SIZE = (128, 128)

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize(IMAGE_SIZE)
    img = np.array(img) / 255.0
    return img.flatten()

@app.route('/', methods=['GET', 'POST'])
def index():
    prediction = None
    if request.method == 'POST':
        # Check if an image file is uploaded
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)

        # Save the uploaded image
        image_path = os.path.join('static', 'uploads', file.filename)
        file.save(image_path)

        # Preprocess the image
        preprocessed_image = preprocess_image(image_path)

        # Get the selected model
        selected_model_name = request.form.get('model')
        selected_model = models[selected_model_name]

        # Make prediction
        prediction = selected_model.predict([preprocessed_image])[0]

        # Clean up uploaded image
        os.remove(image_path)

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
