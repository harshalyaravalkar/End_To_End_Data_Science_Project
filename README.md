# End_To_End_Data_Science_Project

---

# House Rent Prediction using Machine Learning and Streamlit

## Project Description

This project focuses on building a machine learning based house rent prediction system using Python, Scikit-learn, and Streamlit. The main objective of the task was to develop an end-to-end machine learning application capable of predicting monthly house rent prices based on property details provided by users.

House rent prediction is a practical regression problem where machine learning models are trained using historical housing data in order to estimate rental prices for different types of properties. Such systems can help tenants, landlords, and housing platforms better understand property pricing trends and make more informed decisions.

The dataset included several property-related features such as BHK count, property size, furnishing status, city, floor information, bathrooms, and rent amount. During preprocessing, only the most relevant and user-friendly features were selected for prediction in order to simplify the application and improve usability.

The final features used for prediction were:
- BHK
- Size
- City
- Furnishing Status
- Bathroom

This project uses a House Rent dataset containing listings from major Indian cities. After cleaning and preprocessing the data, a Random Forest Regressor was trained to estimate monthly rent based on a few important property features.

The Streamlit interface allows users to enter property details and instantly receive an estimated rent.

The technologies used in this project include:
- Python
- Pandas
- NumPy
- Scikit-learn
- Streamlit

Overall, this project successfully demonstrates a complete machine learning workflow starting from raw dataset preprocessing to deployment of a functional predictive web application.

---

# Project Structure

```bash
CODTECH-Task3-House-Rent-Prediction/
│
├── app.py
├── train_model.py
├── rentdata.csv
├── requirements.txt
└── README.md
```

---

# Requirements

```bash
pip install pandas numpy scikit-learn streamlit
```

---

# How to Run

## Step 1 - Train the Model

```bash
python train_model.py
```

This generates:
- model.pkl
- city_encoder.pkl
- furnish_encoder.pkl

---

## Step 2 - Run the Streamlit Application

```bash
streamlit run app.py
```

After running the command, the application will automatically open in the browser.

---

# Example Inputs

| Feature | Example |
|---|---|
| BHK | 2 |
| Size | 1200 |
| City | Mumbai |
| Furnishing Status | Furnished |
| Bathroom | 2 |

---

# Example Output

```text
Estimated Monthly Rent: ₹ 42,000
```

---

# Screenshots

## Home Page

<img width="1918" height="997" alt="Screenshot 2026-05-20 000335" src="https://github.com/user-attachments/assets/ef837028-2dcb-4153-95fe-9a0280a3c7a0" />


---

## Rent Prediction Result

<img width="853" height="768" alt="Screenshot 2026-05-20 000407" src="https://github.com/user-attachments/assets/a6cf3542-bb0a-443d-a243-f0986514ad01" />


---

# Future Improvements

Possible future enhancements:
- add locality-level prediction
- improve UI design
- deploy online using Streamlit Cloud
- compare multiple regression models
- include maps and visualization dashboards
- add price trend analysis
