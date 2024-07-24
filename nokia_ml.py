import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Load the dataset
df = pd.read_csv("/Users/anav_sobti/Desktop/Hackathons & Ideathons/Nokia Ideathon/integrated_10000_no_null.csv")

# Split the data into features and target
X = df[['altitude', 'location_name', 'region', 'latitude', 'longitude', 'temperature_celsius', 'condition_text', 'pressure_mb', 'humidity']]
y = df['Duct Formation(%)']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define the column transformer with one-hot encoding for categorical data and scaling for numeric data
numeric_features = ['altitude', 'latitude', 'longitude', 'temperature_celsius', 'pressure_mb', 'humidity']
categorical_features = ['location_name', 'region', 'condition_text']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Create a pipeline with the preprocessor and the model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

# Train the model
pipeline.fit(X_train, y_train)

# Predict the target values
y_pred = pipeline.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f'Mean Squared Error: {mse}')
print(f'R^2 Score: {r2}')

# Example input for a new prediction
new_data = pd.DataFrame({
    'altitude': [500],
    'location_name': ['Jhabua'],
    'region': ['Madhya Pardesh'],
    'latitude': [12.34],
    'longitude': [56.78],
    'temperature_celsius': [25],
    'condition_text': ['Clear'],
    'pressure_mb': [1013],
    'humidity': [60]
})

# Predict for the new input data
predicted_value = pipeline.predict(new_data)
print(f'Predicted Duct Formation(%): {predicted_value[0]}')
