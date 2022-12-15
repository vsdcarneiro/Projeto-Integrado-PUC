from typing import Union
from fastapi import FastAPI

from pipeline.predict import predict_quality

app = FastAPI(title='Water Quality')


@app.post("/predict")
def predict_water_quality(data: list):
    try:
        quality = predict_quality(data)
        return {'quality': str(quality)}
    except Exception:
        return {'error': 'It was not possible to classify the water quality.'}
