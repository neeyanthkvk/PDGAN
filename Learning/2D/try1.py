import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
batch_size = 32
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        zoom_range = 0.1,
        horizontal_flip = True)
test_datagen = ImageDataGenerator(rescale = 1./255)
val_datagen = ImageDataGenerator(rescale = 1./255)

train_gen = train_datagen.flow_from_directory('/data/data1/Train', 
        batch_size = 32, 
        target_size=(64, 64),
        class_mode = 'binary')

val_gen = val_datagen.flow_from_directory('/data/data1/Validation',
        batch_size = 32,
        target_size=(64, 64),
        class_mode = 'binary')

test_gen = test_datagen.flow_from_directory('/data/data1/Test',
        batch_size = 32,
        target_size=(64, 64),
        class_mode = 'binary')

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Reshape

model = Sequential()

model.add(Conv2D(64, (5, 5), input_shape = (64, 64, 3), activation='relu'))
model.add(Conv2D(64, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(16, (3, 3), activation = 'relu'))
model.add(Conv2D(16, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.25))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

model.summary()

model.fit_generator(train_gen, 
        steps_per_epoch = 205816 // batch_size, 
        epochs = 15, 
        validation_data = val_gen,
        validation_steps = 44014 // batch_size)

model.save_weights('firsttry.h5')
