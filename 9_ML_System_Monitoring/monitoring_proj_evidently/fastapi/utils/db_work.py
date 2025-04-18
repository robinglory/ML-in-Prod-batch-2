import psycopg2
import os
import pandas as pd

from datetime import datetime

class dbWork():
    def __init__(self):
        
        # Database connection
        self.conn = psycopg2.connect(
            dbname=os.getenv("DB_NAME", "mnistdb"),
            user=os.getenv("DB_USER", "mnistuser"),
            password=os.getenv("DB_PASS", "mnistpass"),
            host=os.getenv("DB_HOST", "postgres_db"),
            port="5432"
        )
        self.cursor = self.conn.cursor()
        print("++++ DB is connected +++++")



    def save_predictions(self,predictions: pd.DataFrame) -> None:



        insert_query = """
            INSERT INTO taxi_predictions (
                passenger_count, trip_distance, fare_amount, total_amount,
                PULocationID, DOLocationID, timestamp, prediction
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """

        for _, row in predictions.iterrows():
            self.cursor.execute(
                    insert_query,
                    (
                        float(row['passenger_count']),
                        float(row['trip_distance']),
                        float(row['fare_amount']),
                        float(row['total_amount']),
                        float(row['PULocationID']),
                        float(row['DOLocationID']),
                        datetime.now(),
                        float(row['prediction'])
                    )
                )


            self.conn.commit()



    def select_table(self,select_query : str):
        df = pd.read_sql_query(select_query, self.conn)
        return df