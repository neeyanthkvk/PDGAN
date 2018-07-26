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
        target_size=(128, 128),
        class_mode = 'binary')

val_gen = val_datagen.flow_from_directory('/data/data1/Validation',
        batch_size = 32,
        target_size=(128, 128),
        class_mode = 'binary')

test_gen = test_datagen.flow_from_directory('/data/data1/Test',
        batch_size = 32,
        target_size=(128, 128),
        class_mode = 'binary')

from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Reshape

model = keras.applications.vgg19.VGG19(include_top=False, weights=None, input_tensor=None, input_shape=(128, 128, 3), classes=2)

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
