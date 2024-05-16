# -*- coding: utf-8 -*-
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import local_binary_pattern, hog
from sklearn.decomposition import PCA
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder

# Assuming each image is resized to 128x128 pixels
img_width, img_height = 128, 128
target_size = (img_width, img_height)

# Function to load and preprocess image data
def load_and_preprocess_data(data_path):
    images = []
    labels = []
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        if os.path.isdir(label_path):
            class_label = label.lower()
            for file in os.listdir(label_path):
                file_path = os.path.join(label_path, file)
                if file.lower().endswith(('.jpg', '.png')):
                    # Load image
                    img = cv2.imread(file_path)
                    img = cv2.resize(img, target_size)
                    images.append(img)
                    labels.append(class_label)
    return np.array(images), np.array(labels)

# Function to extract PCA features
def extract_pca_features(images):
    num_components = min(len(images), images[0].size)
    pca = PCA(n_components=num_components)
    gray_images = [cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) for image in images]
    reshaped_images = np.array([image.reshape(-1) for image in gray_images])
    pca_features = pca.fit_transform(reshaped_images)
    return pca_features

# Function to extract LBP features
def extract_lbp_features(images):
    lbp_radius = 3
    lbp_points = 24
    lbp_features = []
    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        lbp_feature = local_binary_pattern(gray_image, lbp_points, lbp_radius, method='uniform')
        lbp_features.append(lbp_feature.flatten())
    return np.array(lbp_features)

# Function to extract HOG features
def extract_hog_features(images):
    hog_features = []
    for image in images:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        hog_feature = hog(gray_image, orientations=8, pixels_per_cell=(8, 8),
                          cells_per_block=(1, 1), block_norm='L2-Hys')
        hog_features.append(hog_feature.flatten())
    return np.array(hog_features)

# Define data path
data_path = 'D:\\Sign-Language-detection-main\\Sign-Language-detection-main\\Data'

# Load and preprocess images
images, labels = load_and_preprocess_data(data_path)

# Extract PCA features
pca_features = extract_pca_features(images)

# Extract LBP features
lbp_features = extract_lbp_features(images)

# Extract HOG features
hog_features = extract_hog_features(images)

# Concatenate features
features = np.concatenate((pca_features, lbp_features, hog_features), axis=1)

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
num_classes = len(label_encoder.classes_)

# Convert labels to one-hot encoding
encoded_labels = to_categorical(encoded_labels, num_classes)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, encoded_labels, test_size=0.2, random_state=42)

# Build the model
model = Sequential([
    Dense(512, activation='relu', input_shape=(features.shape[1],)),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

# Compile the model
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Early stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)

# Train the model
history = model.fit(X_train, y_train, validation_split=0.2, epochs=20, batch_size=32, callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')

# Plotting the training history
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

# Save the trained model and label encoder classes
model.save('trained_model.h5')
with open('label_encoder_classes.npy', 'wb') as f:
    np.save(f, label_encoder.classes_)

