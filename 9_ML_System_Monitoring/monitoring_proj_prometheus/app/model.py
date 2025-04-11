from tensorflow.keras.models import load_model
import numpy as np
import os
print(os.getcwd())

model = load_model("./model/mnist_model.h5")

def predict_digit(image: np.ndarray):
    image = image.reshape(1, 28, 28)
    image = image / 255.0
    prediction = model.predict(image)
    print("predicted : ",prediction.argmax())
    return prediction.argmax()
