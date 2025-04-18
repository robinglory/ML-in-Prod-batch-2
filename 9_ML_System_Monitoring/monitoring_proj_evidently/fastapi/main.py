from fastapi import FastAPI, UploadFile, File
import numpy as np
from PIL import Image
import io
import os
from uuid import uuid4
import psycopg2
from model import predict_digit,predict_digits
from sklearn.decomposition import PCA

from typing import List
import pandas as pd
import evidently
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



df_reference = pd.read_csv("/fastapi/data/reference_data.csv")




app = FastAPI()


# Database connection
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME", "mnistdb"),
    user=os.getenv("DB_USER", "mnistuser"),
    password=os.getenv("DB_PASS", "mnistpass"),
    host=os.getenv("DB_HOST", "postgres_db"),
    port="5432"
)
cursor = conn.cursor()





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
    
    df_reference
    
    
    image = Image.open(io.BytesIO(contents)).convert("L").resize((28, 28))
    img_array = np.array(image)
    print("img_array : ",img_array.shape)
    
    digit = predict_digit(img_array)

    return {"prediction": int(digit)}
