import time
import flwr as fl
import tensorflow as tf
from shared.model import build_model
from shared.data_utils import load_data_for_client

import os
client_id = int(os.environ.get("CLIENT_ID", 0))
(x_train, y_train), (x_test, y_test) = load_data_for_client(client_id)


model = build_model()

class FlowerClient(fl.client.NumPyClient):
    def get_parameters(self):
        return model.get_weights()

    def fit(self, parameters, config):
        model.set_weights(parameters)
        model.fit(x_train, y_train, epochs=5, batch_size=32, verbose=0)
        return model.get_weights(), len(x_train), {}

    def evaluate(self, parameters, config):
        model.set_weights(parameters)
        loss, acc = model.evaluate(x_train, y_train, verbose=0)

        y_set = set([ int(x) for x in y_train])
        print(f"{client_id} for y_set : ",y_set)
        print(f"{client_id} for acc : ",acc)


        return loss, len(x_train), {"accuracy": acc}
    
fl.client.start_numpy_client(
    server_address="server:8080",
    client=FlowerClient()
)


