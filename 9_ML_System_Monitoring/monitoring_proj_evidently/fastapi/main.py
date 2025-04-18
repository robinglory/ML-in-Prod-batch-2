from fastapi import FastAPI, UploadFile, File, Body
from fastapi.responses import (
    HTMLResponse,
    JSONResponse,
    Response,
    FileResponse
)


import numpy as np
from PIL import Image
import io
import os
import joblib
from uuid import uuid4
from model import predict_digit,predict_digits
from sklearn.decomposition import PCA
from utils.data import load_reference_data,load_taxi_current_data
from typing import List
import pandas as pd
import evidently
from schemas import taxi_model
from utils.db_work import dbWork


print(evidently.__version__)


from evidently.report import Report
from evidently.metrics import ColumnDriftMetric
from evidently.metric_preset import DataDriftPreset



report = Report(
    metrics=[
            ColumnDriftMetric(column_name='mean_pixel'),
            ColumnDriftMetric(column_name='std_pixel') ,
            ColumnDriftMetric(column_name='max_pixel'),
            ColumnDriftMetric(column_name='max_pixel'),
             DataDriftPreset()]
)



app = FastAPI()
db_obj = dbWork()


@app.get('/')
def index() -> HTMLResponse:
    return HTMLResponse('<h1><i>Monitoring Proj with Evidently </i></h1>')


def get_pca_data(_pred_result_:np.array):
    pca = PCA(n_components=5)
    features_reduced = pca.fit_transform(_pred_result_)
    return features_reduced


def extract_image_stats(images):
    stats = {
        "mean_pixel": np.mean(images),
        "std_pixel": np.std(images),
        "min_pixel": np.min(images),
        "max_pixel": np.max(images),
    }
    return stats


def get_new_df(input_image_arr:np.array):
    _pred_test = predict_digits(input_image_arr)
    test_image_data = extract_image_stats(input_image_arr)


    features_reduced_new = get_pca_data(_pred_test)
    df_new = pd.DataFrame(features_reduced_new, columns=[f'pca_{i}' for i in range(5)])
    df_new['target'] = 0  # true labels for the new data
    df_new['timestamp'] = pd.to_datetime('now')
    df_new['mean_pixel'] = test_image_data['mean_pixel']
    df_new['std_pixel'] = test_image_data['std_pixel']
    df_new['min_pixel'] = test_image_data['max_pixel']
    df_new['max_pixel'] = test_image_data['max_pixel']
    return df_new


@app.post("/predict2")
async def predict2(files: List[UploadFile] = File(...)):
    images = []
    for file in files:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))
        img_array = np.array(image)
        images.append(img_array)

    images = np.array(images)
    print("shape : ",images.shape)
    df_new = get_new_df(images)
    
    report = Report(
    metrics=[
            ColumnDriftMetric(column_name='mean_pixel'),
            ColumnDriftMetric(column_name='std_pixel') ,
            ColumnDriftMetric(column_name='max_pixel'),
            ColumnDriftMetric(column_name='max_pixel'),
             DataDriftPreset()]
    )


    df_reference = load_reference_data(file_name="reference_data.csv")

    report.run(reference_data=df_reference, current_data=df_new)
    report_id = uuid4().hex
    report_path = f"/fastapi/reports/model_report_{report_id}.html"
    report.save_html(report_path)

    return {"report_path": report_path}


@app.post("/predict")
async def predict(file: UploadFile):
    print("...increase request count...")
    print("...before file load...")
    contents = await file.read()
    print("...file loaded...")
    
    image = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))
    img_array = np.array(image)
    print("img_array : ",img_array.shape)
    
    digit = predict_digit(img_array)

    return {"prediction": int(digit)}




@app.post("/predict-taxi-duration")
def predict_taxi(body :  taxi_model = Body(...)):
     # Convert to dict, then to DataFrame
    
    features = pd.DataFrame([body.dict()])

    #print("features : ",features)
    #print(features.shape)
    #print(features.dtypes)
    model_path = "/fastapi/data/taxi_lr_model.pkl"
    new_model = joblib.load(model_path)

    print("new_model : ",new_model)
   
    _preds = new_model.predict(features)
    features["prediction"] = _preds

    #Save predition result
    db_obj.save_predictions(predictions=features)
    
    return JSONResponse(content={'prediction': features.to_json()})




from evidently import ColumnMapping
from utils.reports import build_model_performance_taxi_report, build_target_drift_taxi_report

@app.get('/monitor-model')
def monitor_model_performance(window_size: int = 5) -> FileResponse:
 
    target = "duration_min"
    num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
    cat_features = ["pulocationid", "dolocationid"]


    taxi_query = load_taxi_current_data(window_size)

    current_data: pd.DataFrame = db_obj.select_table(select_query=taxi_query)
    current_data["duration_min"] =current_data['prediction']
    print("current_data : ",current_data.shape)
    print(current_data.columns)


    reference_data = load_reference_data(file_name="taxi_reference_data.parquet")
    print("reference_data : ",reference_data.shape)
    print(reference_data.columns)

    
    column_mapping = ColumnMapping(
            target=target,
            prediction='prediction',
            numerical_features=num_features,
            categorical_features=cat_features
        )
    
    report_path= build_model_performance_taxi_report(reference_data= reference_data,
                                        current_data=current_data,
                                        column_mapping=column_mapping)
            

    print("report_path : ",report_path)
    return FileResponse(report_path)




@app.get('/monitor-target')
def monitor_target_drift(window_size: int = 5) -> FileResponse:
    target = "duration_min"
    num_features = ["passenger_count", "trip_distance", "fare_amount", "total_amount"]
    cat_features = ["pulocationid", "dolocationid"]


    taxi_query = load_taxi_current_data(window_size)

    current_data: pd.DataFrame = db_obj.select_table(select_query=taxi_query)
    current_data["duration_min"] =current_data['prediction']

    reference_data = load_reference_data(file_name="taxi_reference_data.parquet")
    column_mapping = ColumnMapping(
            target=target,
            prediction='prediction',
            numerical_features=num_features,
            categorical_features=cat_features
        )
    

    report_path = build_target_drift_taxi_report(
        reference_data=reference_data,
        current_data=current_data,
        column_mapping=column_mapping
    )
    return FileResponse(report_path)
