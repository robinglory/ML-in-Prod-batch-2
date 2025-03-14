### Test mlflow locally first
```bash
# test mlflow with your custom artifcat store and database
mlflow server  --host 127.0.0.1 --port 8080 --default-artifact-root gs://YOUR_GCP_BUCKETNAME  --backend-store-uri postgresql+psycopg2://DB_USERNAME:DB_PASSWORD@DB_IP:5432/DB_NAME


# set application credentials
gcloud auth application-default login
export GOOGLE_APPLICATION_CREDENTIALS="/Users/tharhtet/Documents/github/ML-in-Prod-batch-2/8_Experiment_Tracking/mlflow_prod/credentials/serviceAccount.json"
gsutil ls gs://ths_mlflow_server/1


mlflow server  --host 127.0.0.1 --port 8080 --default-artifact-root gs://ths_mlflow_server 



## Postgres SQL backend with  psycopg2 driver
mlflow server  --host 127.0.0.1 --port 8080 --backend-store-uri postgresql+psycopg2://postgres:tharhtetpwd@34.46.76.38:5432/postgres


## Postgres SQL backend with  pg8000 driver
mlflow server  --host 127.0.0.1 --port 8080 --backend-store-uri postgresql+pg8000://postgres:tharhtetpwd@34.46.76.38:5432/postgres


## Run fully locally 
mlflow server  --host 127.0.0.1 --port 8080 --default-artifact-root gs://ths_mlflow_server --backend-store-uri postgresql+psycopg2://postgres:tharhtetpwd@34.46.76.38:5432/postgres



```
