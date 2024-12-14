import mlflow
import json
import numpy as np
import pickle

model_path = 'C:/Users/Lenovo/Desktop/0800amist-learning-mlops-master 10/mlartifacts/923788704313154080/8e713c3d2ffb4198a7d3bd7480e6c44b/artifacts/rental_prediction_model/model.pkl'

# Use Pickle to dump the Model
#with open(model_path, 'rb') as f:
    #model = pickle.load(f)

# Using Data as JSON File
#with open('inputs/inputs.json', 'r') as f:
    #user_input = json.load(f)

# User Entries Using JSON
#rooms = user_input['rooms']
#sqft = user_input['sqft']
#user_input_prediction= np.array([[rooms,sqft]])

# Model Predicition
#predicted_rental_price = model.predict(user_input_prediction)
#output = {"Rental Prediction using Built Model": predicted_rental_price[0]}

#print (output)


logged_model =  'C:/Users/Lenovo/Desktop/0800amist-learning-mlops-master 10/mlartifacts/923788704313154080/8e713c3d2ffb4198a7d3bd7480e6c44b/artifacts/rental_prediction_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

# Predict on a Pandas DataFrame.
import pandas as pd

predicted_rental_price = loaded_model.predict(pd.DataFrame([[10,10000]]))
output = {"Rental Prediction using Built Model CNR": predicted_rental_price[0]}

print(output)

