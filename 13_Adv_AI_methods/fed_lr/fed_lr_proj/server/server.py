import flwr as fl
from flwr.server.strategy import FedAvg
from flwr.server import ServerConfig
from flwr.common import ndarrays_to_parameters
from shared.model import build_model
model = build_model()

# Get parameters to initialize global model
parameters = model.get_weights()
parameters = ndarrays_to_parameters(model.get_weights())



from shared.data_utils import load_data_for_client
(x_train, y_train), (x_test, y_test) = load_data_for_client(3)

def evaluate(server_round, parameters, config):
    # Set Weights
    model.set_weights(parameters)

    loss, overall_accuracy = model.evaluate(x_train, y_train)

    print(f"Round {server_round}: loss {loss}, accuracy {overall_accuracy}")


# Define strategy
strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_available_clients=2,
    initial_parameters=parameters,
    evaluate_fn=evaluate,
)



# Start server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)