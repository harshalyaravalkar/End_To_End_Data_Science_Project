import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

# Loading Dataset

df = pd.read_csv("rentdata.csv")

# Cleaning Dataset

df = df[df["BHK"] != 6]

# remove unused columns
df.drop(columns=["Posted On"], inplace=True)

# keeping only useful columns
df = df[["BHK", "Size", "City", "Furnishing Status", "Bathroom", "Rent"]]

# Encoding categorical columns

city_encoder = LabelEncoder()
furnish_encoder = LabelEncoder()

df["City"] = city_encoder.fit_transform(df["City"])
df["Furnishing Status"] = furnish_encoder.fit_transform(df["Furnishing Status"])

# splitting into train and test

X = df.drop("Rent", axis=1)
y = df["Rent"]

xtrain, xtest, ytrain, ytest = train_test_split(X,y,test_size=0.2,random_state=42)

# model training

model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)

model.fit(xtrain, ytrain)

# model evaluation

pred = model.predict(xtest)

error = mean_absolute_error(ytest, pred)

print("Mean Absolute Error:", error)

# Saving model pkl files

with open("model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("city_encoder.pkl", "wb") as f:
    pickle.dump(city_encoder, f)

with open("furnish_encoder.pkl", "wb") as f:
    pickle.dump(furnish_encoder, f)

print("model saved")
