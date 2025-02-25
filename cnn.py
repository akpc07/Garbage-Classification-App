

Skip to content
Using Gmail with screen readers
Enable desktop notifications for Gmail.
   OK  No thanks

Conversations
1.55 GB of 15 GB used
Terms · Privacy · Program Policies
Last account activity: 0 minutes ago
Currently being used in 6 other locations · Details
	
Pause mobile notifications while you're using this device
To pause Chat mobile notifications while you’re active on this device, allow your browser to detect if you’re active or away. Click Continue and then Allow when prompted by your browser.

import os
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from tensorflow.keras.preprocessing.image import load_img, img_to_array, ImageDataGenerator
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization, Conv2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam

# Define paths for each garbage category
garbage_paths = {
    "battery": r"C:\Users\aksha\Desktop\project\archive(2)\garbage-dataset\train\battery",
    "biological": r"C:\Users\aksha\Desktop\project\archive(2)\garbage-dataset\train\biological",
    "cardboard": r"C:\Users\aksha\Desktop\project\archive(2)\garbage-dataset\train\cardboard",
    "clothes": r"C:\Users\aksha\Desktop\project\archive(2)\garbage-dataset\train\clothes",
    "glass": r"C:\Users\aksha\Desktop\project\archive(2)\garbage-dataset\train\glass",
    "metal": r"C:\Users\aksha\Desktop\project\archive(2)\garbage-dataset\train\metal",
    "paper": r"C:\Users\aksha\Desktop\project\archive(2)\garbage-dataset\train\paper",
    "plastic": r"C:\Users\aksha\Desktop\project\archive(2)\garbage-dataset\train\plastic",
    "shoes": r"C:\Users\aksha\Desktop\project\archive(2)\garbage-dataset\train\shoes",
    "trash": r"C:\Users\aksha\Desktop\project\archive(2)\garbage-dataset\train\trash"
}

# Initialize lists to store features and labels
X = []
y = []

# Load and preprocess images
for label, path in garbage_paths.items():
    for filename in os.listdir(path):
        if filename.endswith('.jpg') or filename.endswith('.png'):
            image_path = os.path.join(path, filename)
            img = load_img(image_path, target_size=(128, 128))  # Resize image
            img_array = img_to_array(img) / 255.0  # Normalize pixel values
            X.append(img_array)
            y.append(label)

# Convert lists to numpy arrays
X = np.array(X)
y = np.array(y)

# Use LabelEncoder to encode categorical labels into numerical values
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
num_classes = len(label_encoder.classes_)

# Shuffle the data
X, y_encoded = shuffle(X, y_encoded, random_state=42)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Define a simple CNN model
def create_cnn_model(input_shape, num_classes):
    model = Sequential()
    
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=input_shape))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Conv2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(num_classes, activation='softmax'))

    return model

# Create the model
input_shape = (128, 128, 3)
model = create_cnn_model(input_shape, num_classes)

# Compile the model
model.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Train the model with data augmentation
history = model.fit(datagen.flow(X_train, y_train, batch_size=32), 
                    epochs=50, 
                    validation_data=(X_test, y_test), 
                    verbose=1)

# Evaluate the model
loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy}")

# Save the trained model
model.save(r"C:\Users\aksha\Desktop\project\archive(2)\garbage-dataset\cnn_garbage_classification_model.h5")
cnn_model.py
Displaying cnn_model.py.