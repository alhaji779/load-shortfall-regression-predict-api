"""

    Helper functions for the pretrained model to be used within our API.

    Author: Explore Data Science Academy.

    Note:
    ---------------------------------------------------------------------
    Please follow the instructions provided within the README.md file
    located within this directory for guidance on how to use this script
    correctly.

    Importantly, you will need to modify this file by adding
    your own data preprocessing steps within the `_preprocess_data()`
    function.
    ----------------------------------------------------------------------

    Description: This file contains several functions used to abstract aspects
    of model interaction within the API. This includes loading a model from
    file, data preprocessing, and model prediction.  

"""

# Helper Dependencies
import numpy as np
import pandas as pd
import pickle
import json

def _preprocess_data(data):
    """Private helper function to preprocess data for model prediction.

    NB: If you have utilised feature engineering/selection in order to create
    your final model you will need to define the code here.


    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.

    Returns
    -------
    Pandas DataFrame : <class 'pandas.core.frame.DataFrame'>
        The preprocessed data, ready to be used our model for prediction.
    """
    # Convert the json string to a python dictionary object
    feature_vector_dict = json.loads(data)
    # Load the dictionary as a Pandas DataFrame.
    feature_vector_df = pd.DataFrame.from_dict([feature_vector_dict])

    # ---------------------------------------------------------------
    # NOTE: You will need to swap the lines below for your own data
    # preprocessing methods.
    #
    # The code below is for demonstration purposes only. You will not
    # receive marks for submitting this code in an unchanged state.
    # ---------------------------------------------------------------

    # ----------- Replace this code with your own preprocessing steps --------
    #predict_vector = feature_vector_df[['Madrid_wind_speed','Bilbao_rain_1h','Valencia_wind_speed']]
    df_new = feature_vector_df.copy()
    # Impute the missing values with mean imputation
    df_new.fillna(value = df_new.mean(), inplace=True )
    #drop s/no column
    df_new.drop(['Unnamed: 0'], axis=1, inplace=True, errors='ignore')
    # Make time columns datetime format
    df_new['time'] = pd.to_datetime(df_new['time'], utc=False)
    #Create new features from the time variable
    df_new['year'] = df_new['time'].dt.year
    df_new['month'] = df_new['time'].dt.month
    df_new['day'] = df_new['time'].dt.day
    df_new['hour'] = df_new['time'].dt.hour
    df_new['weekday'] = df_new['time'].dt.weekday
    df_new['week'] = df_new['time'].dt.week
    # remove correlated  temperature features since temp is the aveg of temp_min and temp_max, we will drop temp_max and temp_min
    col = ['Madrid_temp_max','Valencia_temp_max','Barcelona_temp_max','Bilbao_temp_max','Seville_temp_max',
          'Madrid_temp_min','Valencia_temp_min','Barcelona_temp_min','Bilbao_temp_min','Seville_temp_min']
    df_new = df_new.drop(col,axis=1, errors='ignore')
    # set time as the index since we have already broken it down into bits..... To avoid duplicate data
    df_new.set_index('time', inplace=True)
    df_new.sort_index(inplace=True)
    # Convert 'Valencia_wind_deg','Seville_pressure' to numerical via pd.get_dummy
    df_new = pd.get_dummies(df_new,columns=['Valencia_wind_deg','Seville_pressure'],
                                   drop_first=True)
    #Replace outliers with the median
    def replace_outlier(val, mean, std,median):
        if val > mean + 3*std:
            return median
        elif val < mean - 3*std:
            return median
        return val
    for col in df_new.columns:
        if df_new[col].dtype == 'float64':
            mean = df_new[col].mean()
            median = df_new[col].median()
            std_dev = df_new[col].std(axis=0)
            df_new[col] = df_new[col].map(lambda x: replace_outlier(x, mean, std_dev,median))
    # ------------------------------------------------------------------------

    return df_new

def load_model(path_to_model:str):
    """Adapter function to load our pretrained model into memory.

    Parameters
    ----------
    path_to_model : str
        The relative path to the model weights/schema to load.
        Note that unless another file format is used, this needs to be a
        .pkl file.

    Returns
    -------
    <class: sklearn.estimator>
        The pretrained model loaded into memory.

    """
    return pickle.load(open(path_to_model, 'rb'))


""" You may use this section (above the make_prediction function) of the python script to implement 
    any auxiliary functions required to process your model's artifacts.
"""

def make_prediction(data, model):
    """Prepare request data for model prediction.

    Parameters
    ----------
    data : str
        The data payload received within POST requests sent to our API.
    model : <class: sklearn.estimator>
        An sklearn model object.

    Returns
    -------
    list
        A 1-D python list containing the model prediction.

    """
    # Data preprocessing.
    prep_data = _preprocess_data(data)
    # Perform prediction with model and preprocessed data.
    prediction = model.predict(prep_data)
    # Format as list for output standardisation.
    return prediction[0].tolist()
