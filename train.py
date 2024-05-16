# -- coding: utf-8 --
import os
import cv2
import numpy as np
import mediapipe as mp
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# Constants
img_width, img_height = 128, 128
target_size = (img_width, img_height)
data_path = 'D:\\Sign-Language-detection-main\\Sign-Language-detection-main\\Data'  # Update this path to your dataset location

# MediaPipe setup for hand landmarks
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

def load_and_preprocess_images(data_path):
    images, labels = [], []
    for label in os.listdir(data_path):
        label_path = os.path.join(data_path, label)
        for file in os.listdir(label_path):
            file_path = os.path.join(label_path, file)
            if file.lower().endswith(('.jpg', '.png')):
                img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)  # Load as grayscale
                img = cv2.resize(img, target_size)
                img = img.astype(np.uint8)  # Ensure the image is in 8-bit format
                images.append(img)
                labels.append(label)
    return np.array(images).reshape(-1, img_width, img_height, 1), labels

def extract_angles_from_images(images):
    angles = []
    for image in images:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)  # Convert grayscale to RGB
        results = hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                angle = calculate_angle(hand_landmarks.landmark[0],
                                        hand_landmarks.landmark[1],
                                        hand_landmarks.landmark[2])
                angles.append(angle)
        else:
            angles.append(0)  # Default angle if no hand detected
    return np.array(angles)

# Load images and labels
images, labels = load_and_preprocess_images(data_path)

# Extract angles
angles = extract_angles_from_images(images)

# Label encoding and converting to categorical
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)
encoded_labels = to_categorical(encoded_labels)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(images, encoded_labels, test_size=0.2, random_state=42)

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(img_width, img_height, 1)),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(len(label_encoder.classes_), activation='softmax')
])

# Compile and train the model
model.compile(optimizer=Adam(), loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(X_train, y_train, epochs=20, batch_size=32, validation_split=0.2)

# Save Keras model
model.save("keras_model.h5")

# Save labeled data
np.savez("labels.npy", images=images, labels=labels)

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Evaluate the model
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f'Test Accuracy: {test_accuracy * 100:.2f}%')
