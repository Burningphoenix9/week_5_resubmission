#Import all the required modules
import pandas as pd
from pycaret.classification import *
import pickle


# create the funcion probability of churn
def probability_of_churn(df_to_predict):
#     check if the input is a dataframe. 
    if type(df_to_predict) == pd.core.frame.DataFrame:
        # Read the model
        with open('ML_model.pickle', 'rb') as ml_model:
            model = pickle.load(ml_model)
        
        # use the read model to predict churn probability of the dataframe just passed in
        predictions = predict_model(model, data=df_to_predict)
        
        # return a binary output of 1: Yes and 0:No
        return list(predictions['prediction_label'].replace('Yes', 1).replace('No', 0))
        
    else:
        # Return if user passed an object that is not a dataframe
        return 'Please input a dataframe'