# TODO: Import your package, replace this by explicit imports of what you need
from buddy.main import predict
import pandas as pd

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allows all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Endpoint for https://your-domain.com/
@app.get("/")
def root():
    return {
        'message': "Hi, The API is running!"
    }

# Endpoint for https://your-domain.com/predict?input_one=154&input_two=199
@app.get("/predict")
def get_predict(str):
    # For the sake of demonstration, just return the sum of the two inputs and the original input
    # Call the predict method with the DataFrame as argument
    prediction = predict(str)

    return {
        'prediction': prediction,
        'length': len(str)
    }
