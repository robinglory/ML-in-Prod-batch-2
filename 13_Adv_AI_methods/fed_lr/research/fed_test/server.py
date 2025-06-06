"""tharhtet: A Flower / TensorFlow app."""
import flwr as fl
from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from flwr.server.strategy import FedAvg
import tf_data_and_model





# this method is not allow in real world
x_train,y_train_one_hot,x_test,y_test_one_hot  = tf_data_and_model.load_data()
x_test_1, y_test_1  = tf_data_and_model.exclue_class(x_test, y_test_one_hot, excluded_cindex=[1, 3, 7])
x_test_2, y_test_2  = tf_data_and_model.exclue_class(x_test, y_test_one_hot, excluded_cindex=[2, 5, 8])
x_test_3, y_test_3  = tf_data_and_model.exclue_class(x_test, y_test_one_hot, excluded_cindex=[4,6,9])






def evaluate(server_round, parameters, config):
    model = tf_data_and_model.load_model()
    # Set Weights
    model.set_weights(parameters)

    _, accuracy137 = tf_data_and_model.evaluate_model(model, x_test_1,y_test_1)
    _, accuracy258 = tf_data_and_model.evaluate_model(model, x_test_2,y_test_2)
    _, accuracy469 = tf_data_and_model.evaluate_model(model, x_test_3,y_test_3)


    print("test accuracy on [1,3,7]: " ,accuracy137)
    print("test accuracy on [2,5,8]: " ,accuracy258)
    print("test accuracy on [4,6,9]: " ,accuracy469)
    



# Get parameters to initialize global model
parameters = ndarrays_to_parameters(tf_data_and_model.load_model().get_weights())

# Define strategy
strategy = FedAvg(
    fraction_fit=1.0,
    fraction_evaluate=1.0,
    min_available_clients=2,
    initial_parameters=parameters,
    evaluate_fn = evaluate
)

# Start server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=3),
    strategy=strategy,
)

