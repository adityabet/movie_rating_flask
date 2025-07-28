# train_model.py
import pandas as pd
import numpy as np
import joblib
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor

# Load data
df = pd.read_csv("IMDb Movies India.csv", encoding='ISO-8859-1')
df = df.dropna(subset=["Rating"])
df["Genre"] = df["Genre"].fillna("Unknown")
df["Director"] = df["Director"].fillna("Unknown")
df["Actor 1"] = df["Actor 1"].fillna("Unknown")
df = df[["Genre", "Director", "Actor 1", "Rating"]]

# Encode
encoder = OneHotEncoder(handle_unknown='ignore')
X = encoder.fit_transform(df[["Genre", "Director", "Actor 1"]])
y = df["Rating"]

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X, y)

# Save model and encoder
joblib.dump(model, "model.pkl")
joblib.dump(encoder, "encoder.pkl")
