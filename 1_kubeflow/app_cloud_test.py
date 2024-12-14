#Import Libraries
import pickle
from flask import Flask

import pandas as pd
import numpy as np
import json

import logging
import boto3
from botocore.exceptions import ClientError
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import root_mean_squared_error
from sklearn.metrics import r2_score

s3 = boto3.client('s3')
response = s3.list_buckets()

bucket = '<<your-bucket>>'

# Data
local_data_file_path = 'data/rental_1000.csv'
s3_data_file_path = 'data/rental_1000.csv'

# Model
local_model_file_path = 'model/rental_prediction_model.pkl'
s3_model_file_path = 'model/rental_prediction_model.pkl'

# Inputs
local_inputs_file_path = 'inputs/inputs.json'
s3_inputs_file_path = 'inputs/inputs.json'

# Outputs
local_outputs_file_path = 'outputs/outputs.json'
s3_outputs_file_path = 'outputs/outputs.json'

app = Flask(__name__)

def download_file_from_s3(bucket, object_name,local_file_name):

    """Download a file from an S3 bucket

    :param bucket: Bucket to upload to
    :param object_name: S3 object name to download. If not specified then file_name is used
    :param local_file_name: File to download
    :return: True if file was downloaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(local_file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.download_file(bucket, object_name,local_file_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

def upload_file_to_s3(local_file_name, bucket, object_name):
    """Upload a file to an S3 bucket

    :param file_name: File to upload
    :param bucket: Bucket to upload to
    :param object_name: S3 object name. If not specified then file_name is used
    :return: True if file was uploaded, else False
    """

    # If S3 object_name was not specified, use file_name
    if object_name is None:
        object_name = os.path.basename(local_file_name)

    # Upload the file
    s3_client = boto3.client('s3')
    try:
        response = s3_client.upload_file(local_file_name, bucket, object_name)
    except ClientError as e:
        logging.error(e)
        return False
    return True

@app.route("/develop_model")
def develop_model():
  # Load the Dataset
    download_file_from_s3(bucket, s3_data_file_path,local_data_file_path)

  # Create DataFrames using Pandas with Data Frim S3
    df = pd.read_csv(local_data_file_path)

  # Features and Labels
    X = df[['rooms','sqft']].values  # Features
    y = df['price'].values           # Label

  #Split the data for Training and Testing
    X_train, X_test, y_train, y_test = train_test_split(X, y,test_size=0.20,random_state=0)

    # Build the model
    lr = LinearRegression()
    model = lr.fit(X_train, y_train)

    # Use Pickle to dump the Model
    with open(local_model_file_path, 'wb') as f:
        pickle.dump(model,f)

    upload_file_to_s3(local_model_file_path, bucket, s3_model_file_path)
    
    # Root Mean Square Error and Score Checks
    #y_pred = model.predict(X_test)
    #root_mean_squared_error_score = root_mean_squared_error(y_test, y_pred)
    #r2_score_value = r2_score(y_test, y_pred)

    #print("####################################### Started Model Development #############################################")
    #print("Root Mean Squared Error for Built Model:",root_mean_squared_error_score + "Accuary Score for Built Model:",r2_score_value)
    #print("Accuary Score for Built Model",r2_score_value)
    #print("####################################### Concluded Model Development #######################################")

@app.route("/model_predict")
def model_predict():
  # Load the Dataset

    develop_model()

    # Inputs for Model Predicitons are pulled from AWS S3
    download_file_from_s3(bucket, s3_inputs_file_path,local_inputs_file_path)

    # Model for Predicitons is to pulled from AWS S3
    download_file_from_s3(bucket, s3_model_file_path,local_model_file_path)

    with open(local_model_file_path, 'rb') as f:
      model = pickle.load(f)

    # Using Data as JSON File
    with open(local_inputs_file_path, 'rb') as f:
      user_input = json.load(f)

    # User Entries Using JSON
    rooms = user_input['rooms']
    sqft = user_input['sqft']
    user_input_prediction= np.array([[rooms,sqft]])

    # Model Predicition
    predicted_rental_price = model.predict(user_input_prediction)
    output = {"Rental Prediction using Built Model": predicted_rental_price[0]}

    #return output

    with open(local_outputs_file_path, 'w') as f:
      json.dump(output, f)

    upload_file_to_s3(local_outputs_file_path, bucket, s3_outputs_file_path)

    #print("####################################### Prediction Started Using Model #######################################")
    #print("Model Predicted and Results are uploaded to outputs")
    #print("####################################### Prediction Concluded Using Model #######################################")

if __name__ == "__main__":
    app.run('0.0.0.0',5000)




