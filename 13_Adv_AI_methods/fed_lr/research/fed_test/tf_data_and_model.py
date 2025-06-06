import numpy as np
import tensorflow as tf
from tensorflow.keras import layers



def load_model(learning_rate: float = 0.001):
    # Define a simple CNN for CIFAR-10 and set Adam optimizer
    model = tf.keras.Sequential(
        [
            layers.Input(shape=(32, 32, 3)),
            layers.Conv2D(64, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Conv2D(128, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Dropout(0.5),
            layers.Conv2D(256, kernel_size=(3, 3), activation="relu"),
            layers.MaxPooling2D(pool_size=(2, 2)),
            layers.Flatten(),
            layers.Dropout(0.5),
            layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        "adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_data():
    # Load CIFAR-10 dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()

    # Normalize the images to the range [0, 1]
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # Convert labels to one-hot encoding if needed
    y_train_one_hot = tf.keras.utils.to_categorical(y_train, 10)
    y_test_one_hot = tf.keras.utils.to_categorical(y_test, 10)

    # Print dataset shapes
    print("Training data shape:", x_train.shape, y_train.shape)
    print("Testing data shape:", x_test.shape, y_test.shape)

    return (x_train,y_train_one_hot,x_test,y_test_one_hot)


def exclue_class(x_data,y_data, excluded_cindex):
    x_filtered,y_filtered = [],[]
    for cur_x, cur_y in zip(x_data,y_data):
        if np.argmax(cur_y) in excluded_cindex:
            continue
        x_filtered.append(cur_x)
        y_filtered.append(cur_y)
    return np.array(x_filtered),np.array(y_filtered)


def check_cindex(cur_y):
    temp = set()
    for x in cur_y:
        temp.add(np.argmax(x))
    return temp


def evaluate_model(model,cur_x_test,cur_y_test):
    loss, accuracy = model.evaluate(cur_x_test,cur_y_test, verbose=0)
    return loss, accuracy

