from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd

app = FastAPI()
from app.ai_for_finance import my_prediction_function

# Cross-Origin Middleware Resource Sharing (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

@app.get("/predict")
def predict(
    open_price:float,
    high:float,
    low:float,
    close:float,
    volume:int
):
    prediction = my_prediction_function(open_price, high, low, close, volume)
    return {"prediction": int(prediction[0])}

@app.get("/")
def root():
    return {
        'greeting': 'Hello'
    }



@app.get("/predict")
def xg_boost_predict( date_pivot :str = '2023-11-04', optional_user_date:str = '2026-04-01'):

     # Charger le CSV
    df = pd.read_csv("../data_folder/cache/BTC-USD.csv")


    # Convertir la 1re colonne en datetime
    df.iloc[:, 0] = pd.to_datetime(df.iloc[:, 0])

    # String de date à partir duquel tu veux filtrer
    date_str = date_pivot
    # Convertir ce string en datetime
    date_cut = pd.to_datetime(date_str)
    # Garder toutes les lignes à partir de cette date
    sub_df = df[df["date"] >= date_cut]


    prediction_xgb, probability_xgb = xgboost_function(sub_df)

    return {"prediction": int(prediction_xgb[-1]), "probability": float(probability_xgb[-1])}
