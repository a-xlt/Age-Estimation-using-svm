import cv2
import numpy as np
import pandas as pd
from skimage.feature import local_binary_pattern
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import glob
import os


def load_and_preprocess_images(directory_path):
    images = []
    ages = []
    for image_path in glob.glob(os.path.join(directory_path, '*.jpg')):  # Adjust pattern if necessary
        age = int(os.path.basename(image_path).split('A')[1][0:2])  # Adjust based on naming convention
        image = cv2.imread(image_path)
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
        for (x, y, w, h) in faces:
            face_img = gray_image[y:y + h, x:x + w]
            face_img = cv2.resize(face_img, (128, 128))
            images.append(face_img)
            ages.append(age)
    return np.array(images), np.array(ages)


def preprocess_images_for_test(directory_path):
    images = []
    image = cv2.imread(directory_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray_image, 1.3, 5)
    for (x, y, w, h) in faces:
        face_img = gray_image[y:y + h, x:x + w]
        face_img = cv2.resize(face_img, (128, 128))
        images.append(face_img)

    return np.array(images)


def extract_features(images):
    hog_features = []
    lbp_features = []
    for image in images:
        hog_descriptor = cv2.HOGDescriptor(_winSize=(128, 128), _blockSize=(16, 16), _blockStride=(8, 8),
                                           _cellSize=(8, 8), _nbins=9)
        hog = hog_descriptor.compute(image).flatten()
        hog_features.append(hog)

        lbp = local_binary_pattern(image, P=24, R=3, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, 27), range=(0, 26))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        lbp_features.append(hist)

    return np.hstack((hog_features, lbp_features))


def mainFunction(path):
    images, ages = load_and_preprocess_images('archive_2/FGNET/images')
    features = extract_features(images)

    testImage = preprocess_images_for_test(path)
    testImageFeatures = extract_features(testImage)

    X_train, X_test, y_train, y_test = train_test_split(features, ages, test_size=0.1, random_state=42, shuffle=True)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(testImageFeatures)
    X_test_scaled2 = scaler.transform(X_test)
    param_grid = {'C': [0.1, 1, 10, 100], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}
    svr = SVR()
    grid_search = GridSearchCV(svr, param_grid, cv=5)
    grid_search.fit(X_train_scaled, y_train)

    best_svr = grid_search.best_estimator_
    predictions = best_svr.predict(X_test_scaled)
    predictions2 = best_svr.predict(X_test_scaled2)

    return int(predictions), mean_absolute_error(predictions2,y_train)
