import streamlit as st
import tensorflow as tf
import numpy as np
from keras.models import load_model
from PIL import Image
from pymongo import MongoClient
from datetime import datetime
import pandas as pd

# MongoDB connection
client = MongoClient(
    'mongodb+srv://hasaranga2019929:V0UfrRapafLyUVyz@cleansentry.aiughjd.mongodb.net/?retryWrites=true&w=majority&appName=cleansentry')
db = client['cleansentry_database']
collection = db['prediction_results']


# Load model function
def model_prediction(image):
    model = load_model('keras_model.h5')
    image = image.resize((224, 224))  # Resize the image to match the model's input shape
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr]) / 255.0  # Convert pixel values to the range [0, 1]
    predictions = model.predict(input_arr)
    return np.argmax(predictions)


# Generate report function
def generate_report():
    data = list(collection.find())
    df = pd.DataFrame(data)
    report = df.groupby(['city', 'prediction_result']).size().unstack(fill_value=0)
    report = report.apply(lambda x: x / x.sum() * 100, axis=1)
    report = report.round(2).astype(str) + '%'
    return report


# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    image_path = "home_page.jpeg"
    st.image(image_path, use_column_width=True)
    st.markdown("""
    Welcome to the Plant Disease Recognition System! 🌿🔍

    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset 
                can be found on this GitHub repo. This dataset consists of about 87K RGB images of healthy and 
                diseased crop leaves which are categorized into 38 different classes. The total dataset is divided 
                into 80/20 ratios of the training and validation set preserving the directory structure. A new 
                directory containing 33 test images is created later for prediction purposes.
                #### Content
                1. train (70295 images)
                2. test (33 images)
                3. validation (17572 images)

                """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")

    # Dropdown for cities
    city = st.selectbox("Select City",
                        ["Dehiwala", "Mount Lavinia", "Galle", "Matara", "Chilaw", "Kalutara", "Mathugama", "Aluthgama",
                         "Beruwala"])

    # Display selected city
    st.write("Selected City:", city)

    # File uploader for multiple images
    uploaded_files = st.file_uploader("Upload Images", type=["png", "jpg", "jpeg"], accept_multiple_files=True)

    # Progress bar
    progress_bar = st.progress(0)

    # Predict button
    if uploaded_files is not None:
        if st.button("Predict"):
            for i, image in enumerate(uploaded_files):
                try:
                    image_data = Image.open(image)
                    result_index = model_prediction(image_data)
                    class_name = ['Metal', 'Glass', 'Plastic', 'Foam']
                    prediction_result = class_name[result_index]

                    # Save prediction result in MongoDB
                    prediction_data = {
                        "city": city,
                        "prediction_result": prediction_result,
                        "last_updated_time": datetime.now()
                    }
                    collection.insert_one(prediction_data)

                    # Update progress bar
                    progress_bar.progress((i + 1) / len(uploaded_files))
                except Exception as e:
                    st.error(f"An error occurred while processing image: {e}")

    # Generate Report Button
    if st.button("Generate Report"):
        report = generate_report()

        # Add report title
        st.markdown("# Cleansentry Automated Detailed Report for Garbage Classification")

        # Display report table
        st.table(report)

        # Download Report Button
        if st.button("Download Report"):
            filename = f"prediction_report_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.csv"
            report.to_csv(filename)
            st.success(f"Report downloaded as {filename}")
