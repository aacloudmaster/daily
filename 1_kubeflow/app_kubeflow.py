import kfp
from kfp import dsl
from kfp import components
from kfp.compiler import Compiler
from kfp.client import Client
import logging

@dsl.component(base_image='python:3.11', packages_to_install=[
    'numpy', 'fsspec', 's3fs==2024.10.0', 'pandas', 'scikit-learn', 
    'boto3', 'appengine-python-standard'
])
def model_build() -> str:
    """Component to develop the model by training and saving it to S3"""
    try:
        import logging
        import pickle
        import os
        import boto3
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        import json
        # Define S3 paths and local paths
        s3_client = boto3.client(
            's3',
            aws_access_key_id='<<your_aws_access_key_id>>',
            aws_secret_access_key='<<your_aws_secret_key_id>>'
        )

        bucket = "iris-bucket-us-east-2"
        model_s3_path = "model/rental-prediction-model.pkl"
        data_s3_path = 'data/rental_1000.csv'
        local_data_path = 'rental_1000.csv'
        local_model_path = 'rental-prediction-model.pkl'

        # Download data from S3
        s3_client.download_file(bucket, data_s3_path, local_data_path)

        # Load the dataset
        rentalDF = pd.read_csv(local_data_path)

        # Prepare the features and labels
        X = rentalDF[["rooms", "sqft"]].values
        y = rentalDF["price"].values
        
        # Split the dataset into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)
        
        # Train the model
        lr = LinearRegression()
        model = lr.fit(X_train, y_train)

        # Save the model using pickle
        with open(local_model_path, 'wb') as f:
            pickle.dump(model, f)

        # Upload the model to S3
        s3_client.upload_file(local_model_path, bucket, model_s3_path)

        return model_s3_path

    except Exception as e:
        logging.error(f"Error in model development: {e}")
        raise


@dsl.component(base_image='python:3.11', packages_to_install=[
    'numpy', 'fsspec', 's3fs==2024.10.0', 'pandas', 'scikit-learn', 
    'boto3', 'appengine-python-standard'
])
def model_predict(value: str, filename: str = 'result.json'):
    """Component to make predictions using the trained model"""
    try:
        import logging
        import pickle
        import os
        import boto3
        import numpy as np
        import pandas as pd
        from sklearn.linear_model import LinearRegression
        from sklearn.model_selection import train_test_split
        import json
    
        # Define S3 paths and local paths
        s3_client = boto3.client(
            's3',
            aws_access_key_id='<<your_aws_access_key_id>>',
            aws_secret_access_key='<<your_aws_secret_key_id>>'
        )

        bucket = "iris-bucket-us-east-2"
        model_s3_path = "model/rental-prediction-model.pkl"
        input_s3_path = "inputs/inputs.json"
        output_s3_path = "outputs/outputs.json"
        local_model_path = 'rental-prediction-model.pkl'
        local_input_path = 'inputs.json'
        local_output_path = 'outputs.json'

        # Download the model and input data from S3
        s3_client.download_file(bucket, model_s3_path, local_model_path)
        s3_client.download_file(bucket, input_s3_path, local_input_path)

        # Load the model using pickle
        with open(local_model_path, 'rb') as f:
            model = pickle.load(f)

        # Load user input (assumed to be in JSON format)
        with open(local_input_path, 'r') as f:
            user_input = json.load(f)

        # Extract user input values
        rooms = user_input['rooms']
        sqft = user_input['sqft']
        user_input_prediction = np.array([[rooms, sqft]])

        # Make a prediction
        predicted_rental_price = model.predict(user_input_prediction)
        output = {"Rental Prediction using Built Model": predicted_rental_price[0]}

        # Save the prediction to an output JSON file
        with open(local_output_path, 'w') as f:
            json.dump(output, f)
            f.write(str(value))  # Save the result into a file

    except Exception as e:
        logging.error(f"Error in model prediction: {e}")
        raise


@dsl.pipeline(
    name='My first pipeline',
    description='A simple pipeline that computes the rental price prediction.'
)
def my_pipeline():
    model_build_task = model_build()
    model_predict_task = model_predict(value=model_build_task.output, filename='results.json')


# Main execution
if __name__ == "__main__":
    kfp_endpoint = None  # Replace with your Kubeflow Pipelines endpoint if needed
    client = kfp.Client(host=kfp_endpoint)

    experiment_name = "Rental Price Prediction Experiment"
    
    # List and delete previous runs if any
    runs = client.list_runs()
    if runs.runs:
        previous_run_id = runs.runs[0].id  # Assuming the latest run is the first in the list
        logging.info(f"Deleting previous run with ID: {previous_run_id}")
        client.delete_run(run_id=previous_run_id)
    
    # Create a new pipeline run
    client.create_run_from_pipeline_func(
        my_pipeline,
        experiment_name=experiment_name
    )
