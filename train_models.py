import os
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from joblib import dump

IMAGE_SIZE = (128, 128)

def preprocess_image(image_path):
    img = Image.open(image_path)
    img = img.resize(IMAGE_SIZE)
    img = np.array(img) / 255.0
    return img.flatten()

def load_dataset(dataset_directory):
    X = []
    y = []

    # Load training data from the 'Train' folder
    train_directory = os.path.join(dataset_directory, 'Train')
    
    for class_name in ['Real', 'Fake']:
        class_directory = os.path.join(train_directory, class_name)
        for image_file in os.listdir(class_directory):
            image_path = os.path.join(class_directory, image_file)
            image_features = preprocess_image(image_path)
            X.append(image_features)
            y.append(1 if class_name == 'Fake' else 0)

    return np.array(X), np.array(y)

def train_and_save_models(dataset_directory):
    X, y = load_dataset(dataset_directory)
    
    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train KNN model
    knn = KNeighborsClassifier(n_neighbors=5)
    knn.fit(X_train, y_train)
    dump(knn, os.path.join('model', 'knn_model.joblib'))

    # Train SVM model
    svm = SVC()
    svm.fit(X_train, y_train)
    dump(svm, os.path.join('model', 'svm_model.joblib'))

    # Train Random Forest model
    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)
    dump(rf, os.path.join('model', 'rf_model.joblib'))

    print("Models trained and saved.")

if __name__ == '__main__':
    dataset_directory = os.path.join(os.getcwd(), 'Dataset')  # Integrated path to your dataset
    train_and_save_models(dataset_directory)
