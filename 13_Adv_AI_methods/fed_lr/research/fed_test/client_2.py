from flwr.client import NumPyClient, ClientApp
from flwr.common import ndarrays_to_parameters, Context
from flwr.client import Client, ClientApp, NumPyClient
from flwr.server.strategy import FedAvg
from flwr.server import ServerApp, ServerConfig
from flwr.server import ServerAppComponents
import flwr as fl


import numpy as np
import tf_data_and_model 


x_train,y_train_one_hot,x_test,y_test_one_hot  = tf_data_and_model.load_data()




x_train_2, y_train_2 = tf_data_and_model.exclue_class(x_train, y_train_one_hot, excluded_cindex=[2, 5, 8])
print("y_train_2 : ",tf_data_and_model.check_cindex(y_train_2))

x_test_2, y_test_2  = tf_data_and_model.exclue_class(x_test, y_test_one_hot, excluded_cindex=[2, 5, 8])
print("y_test_2 : ",tf_data_and_model.check_cindex(y_test_2))




class FlowerClient(NumPyClient):
    def __init__(self):
        self.model = tf_data_and_model.load_model()
        self.x_train, self.y_train, self.x_test, self.y_test = x_train_2, y_train_2,x_test_2, y_test_2
        self.epochs = 3
        self.batch_size = 32
        self.verbose = 2

    def fit(self, parameters, config):
        self.model.set_weights(parameters)
        self.model.fit(
            self.x_train,
            self.y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=self.verbose,
        )
        return self.model.get_weights(), len(self.x_train), {}

    def evaluate(self, parameters, config):
        self.model.set_weights(parameters)
        loss, accuracy = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        print("client_2 loss : ",loss)
        print("client_2 accuracy : ",accuracy)
        
        return loss, len(self.x_test), {"accuracy": accuracy}


client_2 = FlowerClient()
fl.client.start_numpy_client(
    server_address="localhost:8080",
    client=client_2,
)
client_2.model.save("client_2.h5")