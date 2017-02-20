"""
Steering angle prediction model
"""
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Lambda, ELU, Cropping2D
from keras.layers.convolutional import Convolution2D

from sklearn.model_selection import train_test_split

def generate_training_data_from_df(df, batch_size=128):
    batch_X = np.empty([0, 160, 320, 3])
    batch_y = np.empty([0], "float32")
    used=set()
    while 1:
        for i in range(batch_size):
            k = np.random.randint(len(df))
            while k in used:
                k = np.random.randint(len(df))
            batch_X = np.append(batch_X, [plt.imread("data/original/IMG/{}".format(df["center"][k].split("/")[-1]))], axis=0) 
            batch_y = np.append(batch_y, [df["steering"][k]], axis=0)
            used.add(k)
        yield batch_X, batch_y

def generate_validation_data_from_df(df):
    while 1:
        for k in range(len(df)):      
            X = np.array([plt.imread("data/original/IMG/{}".format(df["center"][k].split("/")[-1]))]) 
            y = np.array([df["steering"][k]])
            yield X, y



def get_model():
    row, col, ch = 160, 320, 3  # camera format

    model = Sequential()
    model.add(Lambda(lambda x: x/255 - 0.5, input_shape=(row, col, ch), output_shape=(row, col, ch)))
    model.add(Cropping2D(cropping=((50,20), (1,1)), input_shape=(row, col, ch)))
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
    model.summary()

    return model

def remove_low_throttle_data(df):
    ind = df['throttle']>.25
    return df[ind].reset_index(drop=True)

def remove_zero_steering_data(df):
    ind = df['steering']!=0
    return df[ind].reset_index(drop=True)

def plot_steering_data(df, length):
    steer_s = np.array(df['steering'],dtype=np.float32)
    t_s = np.arange(len(steer_s))
    x_s = np.array(df['steering'])

    plt.plot(t_s[:length],x_s[:length]);
    plt.xlabel('time')
    plt.ylabel('signal')
    plt.ylim(-1,1);
    plt.show()
    
    

if __name__ == "__main__":
    
    model = get_model()

    dtfrm=pd.read_csv("data/original/driving_log.csv")

    dtfrm = remove_low_throttle_data(dtfrm)
    dtfrm = remove_zero_steering_data(dtfrm)
    print("Number of available samples: {}".format(len(dtfrm)))

    nb_val_samples = int(len(dtfrm)*0.20)
    dfT=dtfrm[:len(dtfrm)-nb_val_samples]
    dfV=dtfrm[len(dfT):]

    
    dfT=dfT.reset_index(drop=True)
    dfV=dfV.reset_index(drop=True)    

    print(dfT.tail())
    print(dfV.tail())

    print("Number of Training samples: {}".format(len(dfT)))
    print("Number of Validation samples: {}".format(len(dfV)))


    #plot_steering_data(dfT, 1000)
    #plot_steering_data(dfV, 1000)
    BATCH_SIZE = 128

    train_gen = generate_training_data_from_df(dfT, batch_size=BATCH_SIZE)
    valid_gen = generate_validation_data_from_df(dfV)


    history = model.fit_generator(train_gen, samples_per_epoch=((len(dfT)//BATCH_SIZE)+1)*BATCH_SIZE, validation_data=valid_gen, nb_epoch=6, nb_val_samples=len(dfV), verbose=1)

    print("Saving model weights and configuration file.")

    model.save("model.h5")   
    model.save_weights("model.keras", True)
    with open('model.json', 'w') as outfile:
      json.dump(model.to_json(), outfile)
    import gc; gc.collect()
