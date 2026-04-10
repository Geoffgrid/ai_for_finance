import os
import pickle

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
