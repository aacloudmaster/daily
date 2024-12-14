#Import Libraries
import mlflow
from mlflow.models import infer_signature

import pickle
import mlflow.models
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

# Load the Dataset
df = pd.read_csv('data/rental_1000.csv')

# Features and Labels
X = df[['rooms','sqft']].values  # Features
y = df['price'].values           # Label

#Split the data for Training and Testing
X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20,random_state=0)

# Build the model
lr = LinearRegression()
model = lr.fit(X_train, y_train)

# Root Mean Square Error and Score Checks
y_pred = model.predict(X_test)
root_mean_squared_error_score = root_mean_squared_error(y_test, y_pred)
r2_score_value = r2_score(y_test, y_pred)

# Set our tracking server uri for logging
mlflow.set_tracking_uri(uri="http://127.0.0.1:8080")

# Set the Experiment
mlflow.set_experiment("rental-prediction-experiment")

# Log the loss metric
mlflow.log_metric("RMSE", root_mean_squared_error_score)
mlflow.log_metric("R2", r2_score_value)

# Log the model
model_info = mlflow.sklearn.log_model(
    sk_model=lr,
    artifact_path="rental_prediction_model"
)