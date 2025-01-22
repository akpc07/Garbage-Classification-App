import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten
from sklearn.model_selection import train_test_split

# Define paths
dataset_path = r"C:\\Users\\HP\\Desktop\\garbage\\garbage-dataset"

# Image preprocessing and augmentation
datagen = ImageDataGenerator(
    rescale=1./255,                # Normalize pixel values to [0, 1]
    shear_range=0.2,               # Apply random shear transformations
    zoom_range=0.2,                # Apply random zoom transformations
    horizontal_flip=True,           # Randomly flip images horizontally
    validation_split=0.2           # Reserve 20% for validation
)

# Training and validation data generators
train_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),       # Resize images to 224x224 (VGG16 input size)
    batch_size=32,
    class_mode='categorical',
    subset='training'             # Set for training data
)

validation_generator = datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'           # Set for validation data
)

# Load the VGG16 model, excluding the top layers
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Add custom classification layers
x = base_model.output
x = Flatten()(x)                 # Flatten the output of the last convolutional block
x = Dense(128, activation='relu')(x)   # Add a fully connected layer with 128 neurons
predictions = Dense(train_generator.num_classes, activation='softmax')(x)  # Output layer for classification

# Define the model
model = Model(inputs=base_model.input, outputs=predictions)

# Freeze the convolutional base to retain pre-trained weights
for layer in base_model.layers:
    layer.trainable = False

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=validation_generator
)

# Evaluate the model on validation data
val_loss, val_acc = model.evaluate(validation_generator)
print(f"Validation Accuracy: {val_acc:.4f}, Validation Loss: {val_loss:.4f}")

# Save the model in Keras format
model_save_path = "vgg16_garbage_model.keras"
model.save(model_save_path)  # Save as a Keras model

print(f"Model saved to {model_save_path}")
