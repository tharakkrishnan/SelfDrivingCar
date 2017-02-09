"""
Steering angle prediction model
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as mpplot

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU
from keras.layers.convolutional import Convolution2D

from sklearn.model_selection import train_test_split

def generate_x_y_from_data_frame(df):
    X = np.empty([0, 160, 320, 3])
    y = np.empty([0], "float32")
    sample = 0
    while 1:
        for i in range(128):
            X = np.append(X, [mpplot.imread("data/"+df["center"][sample]], axis=0) 
            y = np.append(df["steering"][sample])
            sample += 1
        yield (X,y)


def get_model():
    ch, row, col = 3, 160, 320  # camera format
    
    model = Sequential()
    model.add(Lambda(lambda x: x/127.5 - 1.,
              input_shape=(ch, row, col),
              output_shape=(ch, row, col)))
    model.add(Convolution2D(16, 8, 8, subsample=(4, 4), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(32, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(ELU())
    model.add(Convolution2D(64, 5, 5, subsample=(2, 2), border_mode="same"))
    model.add(Flatten())
    model.add(Dropout(.2))
    model.add(ELU())
    model.add(Dense(512))
    model.add(Dropout(.5))
    model.add(ELU())
    model.add(Dense(1))
    model.compile(optimizer="adam", loss="mse")

    return model


if __name__ == "__main__":
      
    df=pd.read_csv("data/driving_log.csv")
    print(df.tail())
    gen = generate_x_y_from_data_frame(df)

    model = get_model()

    model.fit_generator(gen, samples_per_epoch=8000, nb_epoch=10, verbose=1)

    print("Saving model weights and configuration file.")

    
    model.save_weights("model.keras", True)
    with open('model.json', 'w') as outfile:
      json.dump(model.to_json(), outfile)
    import gc; gc.collect()
