import pandas as pd




def load_reference_data(file_name: str = 'taxi_reference_data.parquet') -> pd.DataFrame:

    DATA_REF_DIR = "/fastapi/data/"
    ref_path = f"{DATA_REF_DIR}/{file_name}"
    ref_data = pd.read_parquet(ref_path)
    return ref_data


def load_taxi_current_data(window_size : int) -> pd.DataFrame:
    query = f"""
        SELECT 
            passenger_count, trip_distance, fare_amount,
            total_amount, PULocationID, DOLocationID,
            timestamp, prediction
        FROM taxi_predictions
        ORDER BY timestamp DESC
        LIMIT {window_size}
    """

    return query



