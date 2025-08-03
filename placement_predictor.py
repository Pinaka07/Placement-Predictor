import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
import pickle

# Load the data - the CSV is already properly formatted
plc_data = pd.read_csv("placementdata .csv")

# Split data into X and y
X = plc_data.drop("PlacementStatus", axis=1)
y = plc_data["PlacementStatus"]

# Convert target variable to numerical (0 for NotPlaced, 1 for Placed)
y = y.map({'NotPlaced': 0, 'Placed': 1})

# Convert categorical data to numerical using OneHotEncoder
encoder = OneHotEncoder()
# Fix: Correct syntax for ColumnTransformer with multiple columns
trans = ColumnTransformer([
    ("OneHot", encoder, ["ExtracurricularActivities", "PlacementTraining"])
], remainder='passthrough')

trans_data = trans.fit_transform(X)

# Split data into train and test
X_train, X_test, y_train, y_test = train_test_split(
    pd.DataFrame(trans_data), y, test_size=0.2, random_state=42
)

# Scale the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train the model
model = RandomForestRegressor(random_state=42)
model.fit(X_train_scaled, y_train)
y_preds = model.predict(X_test_scaled)

# Evaluate the model
score = r2_score(y_test, y_preds)
print(f"R² Score: {score}")

# Save the model and transformers
pickle.dump(model, open("placement_prediction.pkl", 'wb'))
pickle.dump(trans, open("transformer.pkl", 'wb'))
pickle.dump(scaler, open("scaler.pkl", 'wb'))

# Load the model for prediction
loaded_model = pickle.load(open("placement_prediction.pkl", 'rb'))
loaded_transformer = pickle.load(open("transformer.pkl", 'rb'))
loaded_scaler = pickle.load(open("scaler.pkl", 'rb'))

# Create input data for prediction (fix: add proper values for all columns)
input_data = pd.DataFrame({
    "StudentID": [9999],  # Fix: Added StudentID
    "CGPA": [9.79],
    "Internships": [0],
    "Projects": [2],
    "Workshops/Certifications": [3],
    "AptitudeTestScore": [80],
    "SoftSkillsRating": [3],
    "ExtracurricularActivities": ["Yes"],  # Fix: Use string values like in training data
    "PlacementTraining": ["Yes"],  # Fix: Use string values like in training data
    "SSC_Marks": [85],  # Fix: Added value
    "HSC_Marks": [82]   # Fix: Added value
})

# Transform input data
input_transformed = loaded_transformer.transform(input_data)
input_scaled = loaded_scaler.transform(input_transformed)

# Make prediction
output = loaded_model.predict(input_scaled)
prediction_value = output[0]
# Convert numeric prediction to categorical result
if prediction_value >= 0.5:
    placement_status = "Placed"
else:
    placement_status = "Not Placed"
print(f"Predicted Placement Status: {placement_status} (Score: {prediction_value:.2f})")