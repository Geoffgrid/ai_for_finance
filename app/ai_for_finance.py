import os
import pickle
import pandas as pd


import __main__
from app.ml_logic.features import build_technical_features


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



def xgboost_prediction_function(df:pd.DataFrame):
    model_xgb_path = os.path.join(ROOT_PATH, 'models', 'xgb_pipeline.pkl')

    __main__.build_technical_features = build_technical_features

    with open(model_xgb_path, 'rb') as file:
        model_xgb = pickle.load(file)
        prediction_xgb = model_xgb.predict(df)
        probability_xgb = model_xgb.predict_proba(df)[:, 1]  # Probabilité de la classe positive

    return prediction_xgb, probability_xgb
