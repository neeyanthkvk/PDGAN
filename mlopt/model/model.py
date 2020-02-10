import pickle
import numpy as np
import tensorflow as tf
from abc import ABC, abstractmethod
from sklearn.ensemble import RandomForestClassifier
from tensorflow.python.ops import math_ops
from keras.models import Sequential
from keras.layers import Dense
import keras.backend as K

class Model(ABC):
    def __init__(self, params):
        super().__init__()

    @abstractmethod
    def create_method(self):
        """Base model created using class parameters.

        Arguments:
            None.

        Returns:
            A model.
        """
        pass

    @abstractmethod
    def train_method(self, model, X_train, y_train):
        """Train the model using the given features and labels.

        Arguments:
            model: The model which will be trained.
            X_train: The features on which to be trained.
            y_train: The labels on which to be trained.

        Returns:
            None.
        """
        pass

    @abstractmethod
    def evaluate_method(self, model, X_test):
        """Evaluate a trained model given a new set of features.

        Arguments:
            model: The model which will be evaluated.
            X_test: The features on which to be evaluated.

        Returns:
            The predicted labels from the given features.
        """
        pass

    @abstractmethod
    def save_method(self, model, save_loc):
        """Save the model to a location.

        Arguments:
            model: The model which to be saved.
            save_loc: The file location where the model will be saved.

        Returns:
            None.
        """
        pass


class RandomForest(Model):
    def __init__(self, params):
        """
        Arguments:
            parameters: List of parameters. [Number of Estimators, Maximum Depth]
        """
        super().__init__(params)
        self.parameters = params

    def create_method(self):
        return RandomForestClassifier(n_samples = self.parameters[0],
                                        max_depth = self.parameters[1])

    def train_method(self, model, X_train, y_train):
        model.fit(X_train, y_train)

    def evaluate_method(self, model, X_test):
        return model.predict(X_test)

    def save_method(self, model, save_loc):
        pickle.dump(model, open(save_loc, "wb"))

class NeuralNetwork(Model):
    def __init__(self, params):
        """
        Arguments:
            parameters: List of parameters. [Number of Estimators, Maximum Depth]
        """
        super().__init__(params)
        self.parameters = params

    def new_binary_crossentropy(self, y_true, y_pred):
        ones = K.ones(K.shape(y_true))
        transformed = tf.math.add(tf.math.multiply(y_true, tf.math.log(y_pred)), tf.math.multiply(tf.math.subtract(ones, y_true), tf.math.log(tf.math.subtract(ones, y_pred))))
        return -1 * K.mean(transformed)

    def create_method(self):
        model = Sequential()
        model.add(Dense(self.parameters[0], activation="relu", input_shape=(864067,)))
        model.add(Dense(self.parameters[1], activation="relu"))
        model.add(Dense(1, activation="softmax"))
        if(self.parameters[2]):
            model.compile(optimizer="adam", loss=self.new_binary_crossentropy, metrics=["accuracy"])
        else:
            model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
        return model

    def train_method(self, model, X_train, y_train):
        model.fit(X_train, y_train, epochs = 20, verbose = 1)

    def evaluate_method(self, model, X_test):
        return model.predict(X_test)

    def save_method(self, model, save_loc):
        model_json = model.to_json()
        with open(save_loc + ".json", "w") as json_file:
            json_file.write(model_json)
        model.save_weights(save_loc + ".h5")
