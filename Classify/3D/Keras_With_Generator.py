"""
Regular Deep CNN
44 seconds/epoch for 25 epochs on Tesla K80
Validation Accuracy 94.5%, loss 0.8214
"""
import os
import pickle

import numpy as np
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D
from keras.layers import Flatten, Dropout, Dense, Reshape, BatchNormalization
from keras.layers.advanced_activations import LeakyReLU
from keras.losses import binary_crossentropy

BATCH_SIZE = 16
output_PD_dir = "/data/Imaging/3D/PD"
PD_size = len(os.listdir(output_PD_dir))
output_Control_dir = "/data/Imaging/3D/Control"
Control_size = len(os.listdir(output_Control_dir))
print("TOTAL SIZE: ", PD_size+Control_size)

def load_data(ids):
    X = []
    y = []
    for i in ids:
        loc = ""
        if i < PD_size:
            loc = os.path.join(output_PD_dir, str(i) + ".npy")
        else:
            loc = os.path.join(output_Control_dir, str(i-PD_size) + ".npy")
        x = np.load(loc)
        X.append(x)
        y.append(1 if i < PD_size else 0)
    return np.array(X), np.array(y)


def batch_generator(ids, batch_size=BATCH_SIZE):
    batch = []
    while True:
        np.random.shuffle(ids)
        for i in ids:
            batch.append(i)
            if len(batch) == batch_size:
                yield load_data(batch)
                batch = []


train_inds, test_inds = train_test_split(np.arange(PD_size+Control_size))
train_batch = batch_generator(train_inds)
test_batch = batch_generator(test_inds)

model = Sequential()
model.add(Reshape(target_shape=(128, 128, 128, 1), input_shape=(128, 128, 128)))
model.add(Conv3D(32, kernel_size=(7, 7, 7), strides = (2, 2, 2), padding="same"))
model.add(LeakyReLU())
model.add(Conv3D(32, kernel_size=(7, 7, 7), padding="same"))
model.add(LeakyReLU())
model.add(MaxPooling3D(pool_size=(2, 2, 2), padding="same"))
model.add(Dropout(0.25))

model.add(Conv3D(32, kernel_size=(7, 7, 7), padding="same"))
model.add(LeakyReLU())
model.add(Conv3D(32, kernel_size=(7, 7, 7), padding="same"))
model.add(LeakyReLU())
model.add(MaxPooling3D(pool_size=(2, 2, 2), padding="same"))
model.add(Dropout(0.25))

model.add(Conv3D(128, kernel_size=(5, 5, 5), padding="same"))
model.add(LeakyReLU())
model.add(Conv3D(128, kernel_size=(5, 5, 5), padding="same"))
model.add(LeakyReLU())
model.add(MaxPooling3D(pool_size=(2, 2, 2), padding="same"))
model.add(Dropout(0.25))

model.add(Conv3D(128, kernel_size=(5, 5, 5), padding="same"))
model.add(LeakyReLU())
model.add(Conv3D(128, kernel_size=(5, 5, 5), padding="same"))
model.add(LeakyReLU())
model.add(MaxPooling3D(pool_size=(2, 2, 2), padding="same"))
model.add(Dropout(0.25))

model.add(Conv3D(64, kernel_size=(5, 5, 5), padding="same"))
model.add(LeakyReLU())
model.add(Conv3D(64, kernel_size=(5, 5, 5), padding="same"))
model.add(LeakyReLU())
model.add(MaxPooling3D(pool_size=(2, 2, 2), padding="same"))
model.add(Dropout(0.25))

model.add(Conv3D(64, kernel_size=(5, 5, 5), padding="same"))
model.add(LeakyReLU())
model.add(Conv3D(64, kernel_size=(5, 5, 5), padding="same"))
model.add(LeakyReLU())
model.add(MaxPooling3D(pool_size=(2, 2, 2), padding="same"))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(1, activation='softmax'))
model.compile(loss=binary_crossentropy, optimizer='adam', metrics=['accuracy'])
model.summary()

history = model.fit_generator(train_batch, validation_data=test_batch,
                              epochs=25, verbose=1, shuffle=True,
                              steps_per_epoch=32, validation_steps=8)
pickle.dump(history, open("history.pkl", "wb"))
model.save_weights("7312018_2.h5")
