import streamlit as st
import joblib 
import pandas as pd 
import numpy as np
from datetime import datetime
import base64

# Load pre-trained models for Guntur and Mangalagiri
guntur_temp_model = joblib.load('guntur_temperature.joblib')
guntur_humidity_model = joblib.load('guntur_humidity.joblib')

mangalagiri_temp_model = joblib.load('Mangalagiri_temperature.joblib')
mangalagiri_humidity_model = joblib.load('Mangalagiri_humidity.joblib')


# Sample static data for Guntur and Mangalagiri (for demonstration purposes)
# This would typically come from a CSV or a database.
static_data_guntur = {
    "2018-01-01": {"temp": 28, "humidity": 70},
    "2018-02-01": {"temp": 30, "humidity": 65},
    "2019-01-01": {"temp": 29, "humidity": 68},
    "2020-01-01": {"temp": 31, "humidity": 72},
}

static_data_mangalagiri = {
    "2018-01-01": {"temp": 27, "humidity": 75},
    "2019-01-01": {"temp": 28, "humidity": 73},
    "2021-01-01": {"temp": 30, "humidity": 70},
    "2022-01-01": {"temp": 32, "humidity": 68},
}
st.markdown(
    """
    <style>
    .stApp {
        background-image: url('background.jpg');
        background-size: cover;
        background-position: center;
        background-repeat: no-repeat;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Title of the Streamlit app
st.title("Weather Data")
st.write("The Temperature and Humidity for Guntur and Mangalagiri.")

# City selection widget
city = st.selectbox("Select City", ("Guntur", "Mangalagiri"))

# Date selection widget: Select a date between 2018 to 2024
selected_date = st.date_input("Select Date", min_value=datetime(2018, 1, 1), max_value=datetime(2024, 12, 31), value=datetime(2022, 1, 1))

# Convert the selected date to string format for matching with static data
selected_date_str = selected_date.strftime('%Y-%m-%d')

# Display the selected city and date
st.write("Todays Weather.")
st.write(f"City: {city}")
st.write(f"Date: {selected_date_str}")

# Function to get static data if available
def get_static_data(city, date_str):
    if city == "Guntur" and date_str in static_data_guntur:
        return static_data_guntur[date_str]
    elif city == "Mangalagiri" and date_str in static_data_mangalagiri:
        return static_data_mangalagiri[date_str]
    else:
        return None

# Check for static data first
static_data = get_static_data(city, selected_date_str)

if static_data:
    # If static data exists, display it
    st.write(f"Static Data for {city} on {selected_date_str}:")
    st.write(f"Temperature: {static_data['temp']}°C")
    st.write(f"Humidity: {static_data['humidity']}%")
else:
    # If no static data exists, predict using the model
    if city == "Guntur":
        temp_prediction = guntur_temp_model.predict(np.array([[30.1, 21, 74.6, 19, 10]]))  # Dummy data for prediction
        humidity_prediction = guntur_humidity_model.predict(np.array([[30.1, 21, 74.6, 19, 10]]))  # Dummy data for prediction
    elif city == "Mangalagiri":
        temp_prediction = mangalagiri_temp_model.predict(np.array([[29,21,76.4,20,13.3]]))  # Dummy data for prediction
        humidity_prediction = mangalagiri_humidity_model.predict(np.array([[29,21,76.4,20,13.3]]))  # Dummy data for prediction

    # Display the predicted values
    st.write(f"Temperature for {city} on {selected_date_str}: {temp_prediction[0]:.2f}°C")
    st.write(f"Humidity for {city} on {selected_date_str}: {humidity_prediction[0]:.2f}%")
