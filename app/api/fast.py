from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

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
        date: str,
        close: float,
        high: float,
        low: float,
        open: float,
        volume: int
    ):
    return {
        'date': date,
        'close': close,
        'high': high,
        'low': low,
        'open': open,
        'volume': volume,
        'message': 'Mock API response'
    }


@app.get("/")
def root():
    return {
        'greeting': 'Hello'
    }
