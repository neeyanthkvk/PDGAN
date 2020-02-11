from sklearn.model_selection import train_test_split, KFold
import pandas as pd
import numpy as np
import pickle
import os


class Classifier:
    def __init__(self, inputs, labels, create_method, train_method, evaluate_method, save_method, metrics):
        self.X = inputs
        self.y = labels
        self.create_model = create_method
        self.train = train_method
        self.test = evaluate_method
        self.save = save_method
        self.evaluation_metrics = metrics

    def print_desc(self):
        print("***Printing Model Information*** \n")
        print("**Data Information** \n")
        print("Data Shape: " + self.X.shape)
        if(self.data_split is not None):
            print("Data Split: " + self.data_split)
            print("Training Data Shape: " + self.train_data.shape)
            print("Testing Data Shape: " + self.test_data.shape)
        if(self.num_folds is not None):
            print("Number of Crossvalidation Folds: " + self.num_folds)

        print("**Evaluation Information** \n")
        if(self.training_complete):
            print("Model Performance: " + self.metrics)
            print("Number of Training Epochs Completed: " + self.epochs)

    def split_data(self, split_ratio = 0.2):
        self.data_split = split_ratio
        self.random_seed = np.random.randint(1000)
        self.train_data, self.test_data, self.train_labels, self.test_labels = test_train_split(self.X, self.y,
                                                                                                random_state = self.random_seed,
                                                                                                test_size = self.data_split)
        print("Split Successful: Splitting with the Ratio of: " + self.data_split)

    def set_num_folds(self, num_folds = 8):
        self.num_folds = num_folds

    def single_training_cycle(self, X_train, y_train, X_test, y_test, save = False, save_dir = None):
        self.curr_model = self.create_model()
        self.train(self.curr_model, X_train, y_train)
        y_test_our = self.test(self.curr_model, X_test)
        if(save):
            self.save(self.curr_model, os.path.join(save_dir, "model"))
        return y_test, y_test_our

    def train(self):
        if(self.data_split is None):
            raise Error("Data Split Not Defined")
        if(self.trained):
            raise Error("Already Trainined")
        self.curr_model = self.create_model()
        if(self.num_folds is not None):
            self.training_results = {}
            kf = KFold(n_splits = self.num_folds)
            for counter, train_index, test_index in enumerate(kf.split(self.train_data)):
                X_train_split, X_test_split = self.train_data[train_index], self.train_data[test_index]
                y_train_split, y_test_split = self.train_labels[train_index], self.train_labels[test_index]
                y_acc, y_our = single_training_cycle(X_train_split, y_train_split, X_test_split, y_test_split)
                for metric in self.evaluation_metrics:
                    if metric.__name__ not in self.training_results:
                        self.training_results[metric.__name__] = []
                    print("**DEBUG: SPLIT " + counter + " METRIC " + metric.__name__ + "HAS VALUE " + metric(y_acc, y_our))
                    self.training_results[metric.__name__].append(metric(y_acc, y_our))
        else:
            self.training_results = {}
            y_acc, y_our = single_training_cycle(self.train_data, self.train_labels, self.test_data, self.test_labels)
            for metric in self.evaluation_metrics:
                self.training_results[metric.__name__] = metric(y_acc, y_our)
        self.trained = True;

    def evaluate(self, save = False, save_dir = None):
        if(self.trained is None):
            raise Error("Model not Trained")
        self.testing_results = {}
        y_acc, y_our = single_training_cycle(self.train_data, self.train_labels, self.test_data, self.test_labels, save = save, save_dir = save_dir)
        for metric in self.evaluation_metrics:
            self.testing_results[metric.__name__] = metric(y_acc, y_our)

    def export(self, dir_name):
        d = {}
        if(self.data_split is not None):
            d["data_split"] = [self.data_split]
        if(self.num_folds is not None):
            d["num_folds"] = [self.num_folds]
        if(self.random_seed is not None):
            d["random_seed"] = self.random_seed
        if(self.metrics is not None):
            d["metrics"] = [metric.__name__ for metric in metrics]
        df = pd.DataFrame(data = d)
        df.to_csv(os.path.join(dir_name, "options.csv"))
        with open(os.path.join(dir_name, 'metrics.pkl'), 'wb') as handle:
                pickle.dump(d, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def get_test_metric(self, metric):
        if(self.testing_results is None):
            raise Error("Model not Evaluated")
        if(metric.__name__ not in self.testing_results):
            raise Error("Metric not Evaluated")
        return self.testing_results[metric.__name__]

    def get_training_metric(self, metric):
        if(self.training_results is None):
            raise Error("Model not Trained")
        if(metric.__name__ not in self.training_results):
            raise Error("Metric not Trained")
        return self.testing_results[metric.__name__]
