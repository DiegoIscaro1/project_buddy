# TODO: Import your package, replace this by explicit imports of what you need
from buddy.main import predict
import pandas as pd

import nltk
nltk.download('stopwords')
nltk.download('wordnet')

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

# Endpoint for https://your-domain.com/predict?txt=I feel sad
@app.get("/predict")
def get_predict(txt: str):
    # Call the predict method with the input text
    output = predict(txt)
    return {
        'prediction': output
    }
