from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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
