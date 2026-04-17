import os
import pickle
import pandas as pd

ROOT_PATH = os.path.dirname(os.path.dirname(__file__))

def my_prediction_function(
    open_price,
    high,
    low,
    close,
    volume
):
    model_path = os.path.join(ROOT_PATH, 'models', 'linear_regression_1.pkl')
    with open(model_path, 'rb') as file:
        model = pickle.load(file)
        prediction = model.predict([[open_price, high, low, close, volume]])

    return prediction



def xgboost_function(df:pd.DataFrame):
    model_xgb_path = os.path.join(ROOT_PATH, 'models', 'xgb_pipeline.pkl')
    with open(model_xgb_path, 'rb') as file:
        model_xgb = pickle.load(file)
        prediction_xgb = model_xgb.predict(df)

    return prediction_xgb
