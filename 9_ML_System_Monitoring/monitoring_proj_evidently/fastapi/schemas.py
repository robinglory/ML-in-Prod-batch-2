
from pydantic import BaseModel,Field

class taxi_model(BaseModel):
    passenger_count :float =      1.00
    trip_distance   :float =      4.16
    fare_amount   :float =       23.30
    total_amount  :float =       29.80
    PULocationID   :float =     260.00
    DOLocationID :float =       138.00
