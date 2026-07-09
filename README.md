# End_To_End_Data_Science_Project

---

# House Rent Prediction using Machine Learning and Streamlit

## Project Description

This project focuses on building a machine learning based house rent prediction system using Python, Scikit-learn, and Streamlit. The main objective of the task was to develop an end-to-end machine learning application capable of predicting monthly house rent prices based on property details provided by users.

House rent prediction is a practical regression problem where machine learning models are trained using historical housing data in order to estimate rental prices for different types of properties. Such systems can help tenants, landlords, and housing platforms better understand property pricing trends and make more informed decisions.

For this project, a real-world House Rent dataset was used containing rental listings from multiple Indian cities including:
- Bangalore
- Chennai
- Delhi
- Hyderabad
- Kolkata
- Mumbai

The dataset included several property-related features such as BHK count, property size, furnishing status, city, floor information, bathrooms, and rent amount. During preprocessing, only the most relevant and user-friendly features were selected for prediction in order to simplify the application and improve usability.

The final features used for prediction were:
- BHK
- Size
- City
- Furnishing Status
- Bathroom

The first stage of the project involved data preprocessing and cleaning. Unnecessary columns were removed from the dataset to reduce complexity and improve model performance. Since some columns contained categorical text data such as city names and furnishing status, label encoding was applied to convert these values into numerical form suitable for machine learning algorithms.

After preprocessing, the dataset was divided into training and testing sets using Scikit-learn’s `train_test_split` function. This allowed the model to learn patterns from training data and later evaluate performance using unseen testing data.

For prediction, a `RandomForestRegressor` model was implemented. Random Forest is an ensemble machine learning algorithm that combines multiple decision trees to improve prediction accuracy and reduce overfitting. The model performed well on the structured housing dataset and was able to capture relationships between property features and rental prices.

After training the model, it was saved using Python’s `pickle` module. Model serialization allowed the trained model to be reused directly inside the deployed web application without retraining each time the application runs.

The deployment stage of the project was developed using Streamlit. Streamlit was used to create an interactive and user-friendly web interface where users can easily provide property details and receive rent predictions instantly.

The application interface allows users to enter:
- BHK count
- property size in square feet
- city
- furnishing status
- bathroom count

Dropdown menus, sliders, and input fields were used to make the interface simple and easy to understand. Once the details are entered, the application predicts the estimated monthly rent value.

This project demonstrates several important concepts related to machine learning and deployment, including:
- data preprocessing
- feature selection
- categorical encoding
- regression modeling
- model serialization
- web application deployment

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

# Features of the Project

- real-world housing dataset
- data preprocessing and cleaning
- label encoding for categorical data
- Random Forest regression model
- model serialization using pickle
- interactive Streamlit web application
- real-time rent prediction

---

# Business Applications

This type of prediction system can be useful for:
- tenants comparing rental prices
- landlords estimating rent values
- housing and rental platforms
- property analytics systems
- real estate recommendation platforms

---

# Future Improvements

Possible future enhancements:
- add locality-level prediction
- improve UI design
- deploy online using Streamlit Cloud
- compare multiple regression models
- include maps and visualization dashboards
- add price trend analysis
