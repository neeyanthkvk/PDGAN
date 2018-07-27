import keras
import numpy as np
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
batch_size = 128
train_datagen = ImageDataGenerator(
        zoom_range = 0.1,
        horizontal_flip = True)
val_datagen = ImageDataGenerator()

train_gen = train_datagen.flow_from_directory('/data2/data1/Train', 
        batch_size = 32, 
        class_mode = 'binary')

val_gen = val_datagen.flow_from_directory('/data2/data1/Validation',
        batch_size = 32,
        class_mode = 'binary')

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Reshape

model = Sequential()

model.add(Conv2D(64, (3, 3), input_shape = (None, None, 3), activation='relu'))
model.add(Dropout(0.125))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(Dropout(0.125))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(Dropout(0.125))
model.add(Conv2D(32, (3, 3), activation = 'relu'))
model.add(Dropout(0.125))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(16, (3, 3), activation = 'relu'))
model.add(Dropout(0.25))
model.add(Conv2D(16, (3, 3), activation = 'relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.25))

model.add(GlobalAveragePooling2D())
model.add(Dense(64, activation = 'relu'))
model.add(Dropout(0.375))
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))

model.compile(loss='binary_crossentropy',
        optimizer='adam',
        metrics=['accuracy'])

model.summary()

stopper = keras.callbacks.EarlyStopping(monitor='loss', patience=5, verbose = 1)
board = keras.callbacks.TensorBoard(log_dir='./logs', histogram_freq=0, batch_size=32, write_graph=True, write_grads=False, write_images=False, embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None, embeddings_data=None)

model.fit_generator(train_gen, 
        steps_per_epoch = 238500 // batch_size, 
        epochs = 100, 
        validation_data = val_gen,
        validation_steps = 55841 // batch_size,
        callbacks = [board, stopper])

model.save_weights('firsttry.h5')
