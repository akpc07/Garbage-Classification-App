import os
import numpy as np
import pandas as pd
from PIL import Image
import streamlit as st
from tensorflow.keras.models import load_model
import warnings
import base64

# Suppress warnings
warnings.filterwarnings("ignore")

# File for storing predictions
predictions_file = 'predictions.xlsx'

# Class names for the garbage categories
class_names = [
    "battery", "biological", "cardboard",
    "clothes", "glass", "metal", "paper", "plastic", "shoes", "trash"
]

# Disposal methods for each category
disposal_methods = {
    "clothes": "Donate or recycle at a textile recycling facility.",
    "plastic": "Recycle at a plastic recycling facility.",
    "metal": "Recycle at a metal recycling center.",
    "paper": "Recycle with paper waste.",
    "battery": "Dispose at a hazardous waste facility.",
    "biological": "Compost if possible or dispose of in a biohazard bag.",
    "cardboard": "Recycle with cardboard materials.",
    "glass": "Recycle at a glass recycling center.",
    "shoes": "Donate or dispose of with textiles.",
    "trash": "Dispose of in regular trash bins."
}

# Function to save predictions
def save_prediction(location, prediction, user_type, model_name):
    if os.path.exists(predictions_file):
        df = pd.read_excel(predictions_file)
    else:
        df = pd.DataFrame(columns=['Location', 'Prediction', 'User Type', 'Model Used'])
    
    new_data = pd.DataFrame({
        'Location': [location],
        'Prediction': [prediction],
        'User Type': [user_type],
        'Model Used': [model_name]
    })
    df = pd.concat([df, new_data], ignore_index=True)
    df.to_excel(predictions_file, index=False)

# Function to load predictions
def load_predictions():
    if os.path.exists(predictions_file):
        df = pd.read_excel(predictions_file)
        return df
    else:
        return pd.DataFrame(columns=['Location', 'Prediction', 'User Type', 'Model Used'])

# Function to delete individual predictions
def delete_individual_prediction(index):
    if os.path.exists(predictions_file):
        df = pd.read_excel(predictions_file)
        if 0 <= index < len(df):
            df = df.drop(index)
            df.to_excel(predictions_file, index=False)

# Function to delete all predictions
def delete_all_predictions():
    if os.path.exists(predictions_file):
        os.remove(predictions_file)

# Function to make a prediction using the selected model
def make_prediction(uploaded_file, model):
    try:
        # Load and preprocess image
        img = Image.open(uploaded_file).convert('RGB')

        # Check model input shape to determine the correct resizing
        required_size = model.input_shape[1:3]  # (height, width)
        img = img.resize(required_size)

        # Convert image to numpy array and normalize
        img_array = np.array(img, dtype=np.float32) / 255.0  # Normalize pixel values to [0, 1]
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension (1, height, width, channels)

        # Get predictions
        predictions = model.predict(img_array)

        # Get the index of the class with the maximum confidence
        predicted_class_index = np.argmax(predictions[0])

        # Get the predicted class name
        predicted_class_name = class_names[predicted_class_index]

        return predicted_class_name  # Only return the predicted class name

    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return None

def load_models():
    models = {}  # Initialize as a dictionary
    try:
        # Load and compile models only once
        vgg16_path = "C:\\Users\\HP\\Desktop\\garbage\\second.keras"
        cnn_path = "C:\\Users\\HP\\Desktop\\garbage\\garbage_classifier.h5"

        if os.path.exists(vgg16_path):
            models['vgg16'] = load_model(vgg16_path)
            models['vgg16'].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        if os.path.exists(cnn_path):
            models['cnn'] = load_model(cnn_path)
            models['cnn'].compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        return models if models else None
    except Exception as e:
        st.error(f"Error loading models: {e}")
        return None

# Function to set background image
def set_background(image_path):
    if os.path.exists(image_path):
        with open(image_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode()
        st.markdown(
            f"""
            <style>
            .stApp {{
                background: url(data:image/jpeg;base64,{encoded_image});
                background-size: cover;
                background-repeat: no-repeat;
                background-position: center;
            }}
            </style>
            """,
            unsafe_allow_html=True
        )
    else:
        st.error("Failed to load background image.")

# Main app function
def main():
    image_path = "C:\\Users\\HP\\Desktop\\garbage\\background.png" 
    set_background(image_path)

    st.title("Garbage Classification App")

    models = load_models()
    if not models:
        st.error("No models were loaded. Exiting the application.")
        return

    if 'logged_in' not in st.session_state:
        st.session_state['logged_in'] = False
        st.session_state['user_type'] = None

    if not st.session_state['logged_in']:
        st.subheader("Login Page")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "admin123":
                st.session_state['logged_in'] = True
                st.session_state['user_type'] = 'admin'
                st.success("Logged in as Admin")
            elif username == "user" and password == "user123":
                st.session_state['logged_in'] = True
                st.session_state['user_type'] = 'user'
                st.success("Logged in as User")
            else:
                st.error("Invalid credentials. Please try again.")
    else:
        if st.session_state['user_type'] == 'user':
            user_dashboard(models)
        elif st.session_state['user_type'] == 'admin':
            admin_dashboard(models)

        if st.button("Logout"):
            st.session_state['logged_in'] = False
            st.session_state['user_type'] = None
            st.success("You have been logged out.")

# User dashboard for uploading and classifying images
def user_dashboard(models):
    st.header("User Dashboard")

    location = st.text_input("Enter your location")
    uploaded_file = st.file_uploader("Upload a garbage image", type=["jpg", "jpeg", "png"])

    # Allow the user to select a model
    selected_model = st.selectbox("Select a model for classification", options=['vgg16', 'cnn'])

    if st.button("Classify"):
        if uploaded_file and location:
            model = models[selected_model]
            predicted_class_name = make_prediction(uploaded_file, model)
            if predicted_class_name:
                st.success(f"Predicted Class: {predicted_class_name}")
                st.write(f"Disposal Method: {disposal_methods[predicted_class_name]}")

                # Save the prediction
                save_prediction(location, predicted_class_name, st.session_state['user_type'], selected_model)
        else:
            st.error("Please upload an image and enter your location.")

# Admin dashboard for viewing and managing predictions
def admin_dashboard(models):
    st.header("Admin Dashboard")

    predictions_df = load_predictions()

    if not predictions_df.empty:
        st.subheader("Prediction Records")
        st.dataframe(predictions_df)

        if st.button("Delete All Predictions"):
            delete_all_predictions()
            st.success("All predictions have been deleted.")

        # Option to delete individual predictions
        index_to_delete = st.number_input(
            "Enter the index of the prediction to delete", min_value=0, max_value=len(predictions_df) - 1, value=0
        )
        if st.button("Delete Prediction"):
            delete_individual_prediction(index_to_delete)
            st.success(f"Prediction at index {index_to_delete} has been deleted.")

    else:
        st.warning("No predictions available.")

# Run the app
if __name__ == "__main__":
    main()
