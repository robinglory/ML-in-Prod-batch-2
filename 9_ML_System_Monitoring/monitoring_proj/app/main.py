from fastapi import FastAPI, UploadFile
import numpy as np
from PIL import Image
import io
from model import predict_digit
from metrics import REQUEST_COUNT, INFERENCE_LATENCY
from prometheus_client import start_http_server
from fastapi.responses import PlainTextResponse
import prometheus_client

app = FastAPI()

start_http_server(8001)  # Expose Prometheus metrics

@app.post("/predict")
async def predict(file: UploadFile):
    print("...increase request count...")
    REQUEST_COUNT.inc()
    print("...before file load...")
    contents = await file.read()
    print("...file loaded...")
    with INFERENCE_LATENCY.time():
        image = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))
        img_array = np.array(image)
        digit = predict_digit(img_array)
        return {"prediction": int(digit)}

@app.get("/metrics", response_class=PlainTextResponse)
def metrics():
    return prometheus_client.generate_latest()
