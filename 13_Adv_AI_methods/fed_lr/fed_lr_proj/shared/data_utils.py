import tensorflow as tf

import numpy as np
import tensorflow as tf
import os

def load_data_for_client(client_id: int):
    # Load MNIST dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0
    x_train = x_train[..., np.newaxis]
    x_test = x_test[..., np.newaxis]

    # Define class splits
    label_map = {
        0: [0, 1, 2, 3],       # client-1
        1: [3, 4, 5],          # client-2
        2: [6, 7, 8, 9],       # client-3
        3: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9], # for server
    }

    allowed_labels = label_map[client_id]

    # Filter training data
    train_mask = np.isin(y_train, allowed_labels)
    test_mask = np.isin(y_test, allowed_labels)

    return (x_train[train_mask], y_train[train_mask]), (x_test[test_mask], y_test[test_mask])

