import keras
import numpy as np
import keras.backend as K
from sklearn.model_selection import train_test_split

# Model Creation
from keras.models import Sequential
from keras.layers import Conv3D, MaxPooling3D, GlobalMaxPooling3D, GlobalAveragePooling3D
from keras.layers import ConvLSTM2D
from keras.layers import Conv2D, MaxPooling2D, GlobalMaxPooling2D, GlobalAveragePooling2D
from keras.layers import Flatten, Dropout, Dense, Permute, BatchNormalization
from keras.optimizers import Adam
from keras.losses import categorical_crossentropy

model = Sequential()
model.add(Conv3D(filters=32, kernel_size=(7, 7, 7), padding = 'same', input_shape = (176, 256, 240, 1)))
model.add(keras.layers.LeakyReLU(alpha=0.125))
model.add(Conv3D(filters=32, kernel_size=(7, 7, 7), padding = 'same'))
model.add(keras.layers.LeakyReLU(alpha=0.125))
model.add(MaxPooling3D((4, 4, 4)))

model.add(Conv3D(filters=64, kernel_size=(5, 5, 5), padding = 'same'))
model.add(keras.layers.LeakyReLU(alpha=0.125))
model.add(Conv3D(filters=64, kernel_size=(5, 5, 5), padding = 'same'))
model.add(keras.layers.LeakyReLU(alpha=0.125))
model.add(MaxPooling3D((2, 2, 2)))


model.add(ConvLSTM2D(filters=32, kernel_size=(5, 5), padding = 'same', recurrent_activation='relu', return_sequences=True)) 
model.add(keras.layers.LeakyReLU(alpha=0.125))

model.add(ConvLSTM2D(filters=64, kernel_size=(5, 5), padding = 'same', recurrent_activation='relu', return_sequences=True)) 
model.add(keras.layers.LeakyReLU(alpha=0.125))
model.add(ConvLSTM2D(filters=16, kernel_size=(5, 5), padding = 'same', recurrent_activation='relu', return_sequences=False)) 
model.add(keras.layers.LeakyReLU(alpha=0.125))

model.add(Conv2D(filters=64, kernel_size=(7,7), padding = 'same'))
model.add(keras.layers.LeakyReLU(alpha=0.125))
model.add(Conv2D(filters=64, kernel_size=(7,7), padding = 'same'))
model.add(keras.layers.LeakyReLU(alpha=0.125))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(filters=64, kernel_size=(5,5), padding = 'same'))
model.add(keras.layers.LeakyReLU(alpha=0.125))
model.add(Conv2D(filters=64, kernel_size=(5,5), padding = 'same'))
model.add(keras.layers.LeakyReLU(alpha=0.125))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(filters=32, kernel_size=(5,5), padding = 'same'))
model.add(keras.layers.LeakyReLU(alpha=0.125))
model.add(Conv2D(filters=32, kernel_size=(5,5), padding = 'same'))
model.add(keras.layers.LeakyReLU(alpha=0.125))
model.add(MaxPooling2D((2,2)))

model.add(Conv2D(filters=32, kernel_size=(3,3), padding = 'same'))
model.add(keras.layers.LeakyReLU(alpha=0.125))
model.add(Conv2D(filters=32, kernel_size=(3,3), padding = 'same'))
model.add(keras.layers.LeakyReLU(alpha=0.125))
model.add(MaxPooling2D((2,2)))

model.add(Flatten())
model.add(BatchNormalization())
model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation = 'relu'))
model.add(Dense(2, activation='softmax'))
model.compile(loss=categorical_crossentropy, optimizer='adam', metrics=['accuracy'])
model.summary()

# Data Loading
y = np.zeros((466+148, 2))
x = np.concatenate([np.load("/data2/3D/PD.npy"), np.load("/data2/3D/Control.npy")])
print("LOADED")
for i in range(466):
    y[i][0] = 1
for i in range(466, 466+148):
    y[i][1] = 1
print("MADE Y")
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=42)
x = 0
y = 0



# Model Testing
history = model.fit(x_train, y_train, validation_data = (x_test, y_test), batch_size=1, epochs = 25, verbose = 1, shuffle = True)
import pickle
pickle.dump(history, open("history.pkl", "wb"))
model.save_weights("7312018.h5")

