import streamlit as st
import pandas as pd
import pickle
import os

# Set a wide layout for the app for better visual experience
st.set_page_config(layout="wide")

# ---- File Loading and Model Initialization ----
# The model file is assumed to be in the same directory as this script.
model_file = "best_random_forest.pkl"

# Check if the model file exists before trying to load it.
if not os.path.exists(model_file):
    st.error(f"The model file '{model_file}' was not found. Please ensure it is in the same directory as this script.")
    st.stop() # Stop the app execution if the model file is missing

# Load the pre-trained machine learning pipeline
# @st.cache_resource decorator caches the model to prevent reloading on every user interaction
@st.cache_resource
def load_model():
    """Loads the pre-trained model pipeline from the pickle file."""
    try:
        with open(model_file, "rb") as f:
            pipeline = pickle.load(f)
        return pipeline
    except Exception as e:
        st.error(f"Failed to load the model. Error: {e}")
        return None

pipeline = load_model()

# ---- UI Elements and Input Widgets ----
st.title("🏡 Indian House Price Predictor")
st.markdown("Enter the property details below to get an estimated price.")

# Use columns for a more organized and clean layout
col1, col2, col3 = st.columns(3)

# First column for numeric inputs
with col1:
    st.subheader("Property Dimensions")
    area = st.number_input("Area (in square feet)", min_value=1.0, value=5000.0, step=100.0)
    bedrooms = st.number_input("Number of Bedrooms", min_value=1, max_value=10, value=3)
    bathrooms = st.number_input("Number of Bathrooms", min_value=1, max_value=5, value=2)

# Second column for categorical inputs
with col2:
    st.subheader("Key Features")
    stories = st.number_input("Number of Stories", min_value=1, max_value=5, value=1)
    parking = st.number_input("Parking Spaces", min_value=0, max_value=5, value=1)
    mainroad = st.radio("Main Road Access", ["yes", "no"])

# Third column for more categorical inputs
with col3:
    st.subheader("Additional Amenities")
    guestroom = st.radio("Guest Room", ["yes", "no"])
    basement = st.radio("Basement", ["yes", "no"])
    hotwaterheating = st.radio("Hot Water Heating", ["yes", "no"])
    airconditioning = st.radio("Air Conditioning", ["yes", "no"])
    prefarea = st.radio("Preferred Area", ["yes", "no"])
    furnishingstatus = st.selectbox(
        "Furnishing Status",
        ["furnished", "semi-furnished", "unfurnished"]
    )

# ---- Prediction Logic ----
# This section runs only when the button is clicked
st.markdown("---")
if st.button("Predict Price", help="Click to predict the house price"):
    if pipeline:
        try:
            # Create a DataFrame from user inputs, matching the model's expected format
            input_data = pd.DataFrame({
                "area": [area],
                "bedrooms": [bedrooms],
                "bathrooms": [bathrooms],
                "stories": [stories],
                "parking": [parking],
                "mainroad": [mainroad],
                "guestroom": [guestroom],
                "basement": [basement],
                "hotwaterheating": [hotwaterheating],
                "airconditioning": [airconditioning],
                "prefarea": [prefarea],
                "furnishingstatus": [furnishingstatus]
            })

            # Make the prediction using the loaded pipeline
            prediction = pipeline.predict(input_data)[0]

            # Display the result to the user
            st.success(f"### 💰 Estimated Price: ₹ {prediction:,.0f}")
            st.balloons() # Visual celebration for a successful prediction

        except Exception as e:
            st.error(f"An error occurred during prediction. Please check your inputs. Error: {e}")
    else:
        st.warning("The model is not loaded. Please try refreshing the page or check the file path.")
