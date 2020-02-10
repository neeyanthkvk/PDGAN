from abc import ABC, abstractmethod

class Model(ABC):
    def __init__(self, params):
        super().__init__(params)

    @abstractmethod
    def create_method():
        pass

    @abstractmethod
    def train_method():
        pass

    @abstractmethod
    def evaluate_method():
        pass

    @abstractmethod
    def save_method():
        pass
