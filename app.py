import streamlit as st
import pickle
import numpy as np

# ---------------- LOAD ----------------

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

with open("city_encoder.pkl", "rb") as f:
    city_encoder = pickle.load(f)

with open("furnish_encoder.pkl", "rb") as f:
    furnish_encoder = pickle.load(f)

# ---------------- UI ----------------

st.set_page_config(page_title="House Rent Prediction")

st.title("🏠 House Rent Prediction")

st.write("Enter property details to estimate monthly rent.")

# ---------------- INPUTS ----------------

bhk = st.number_input("BHK", min_value=1, max_value=10, value=2)

size = st.number_input("Size (in sq ft)", min_value=100, max_value=10000, value=1200)

bathroom = st.number_input("Bathrooms", min_value=1, max_value=10, value=2)

city = st.selectbox(
    "City",
    ["Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai"]
)

furnishing = st.selectbox(
    "Furnishing Status",
    ["Furnished", "Semi-Furnished", "Unfurnished"]
)

# ---------------- ENCODE ----------------

city_encoded = city_encoder.transform([city])[0]

furnishing_encoded = furnish_encoder.transform([furnishing])[0]

# ---------------- PREDICT ----------------

if st.button("Predict Rent"):

    features = np.array([[
        bhk,
        size,
        city_encoded,
        furnishing_encoded,
        bathroom
    ]])

    prediction = model.predict(features)[0]

    st.success(f"Estimated Monthly Rent: ₹ {prediction:,.0f}")