from skopt import gp_minimize
from mlopt.classifier import Classifier

class Optimizer:
    def __init__(self, min_func, model_class, parameters, X, y):
        """
        The Model Class must have a create_method, train_method, evaluate_method, and save_method
        """
        self.error = min_func
        self.model = model_class
        self.parameters = parameters
        self.X = X
        self.y = y

    def single_iter(self, X, y, save = False, save_dir = None):
        def inner_eval(params):
            model = model_class(params)
            classifier = Classifier(X, y, model.create_method, model.train_method,
                                    model.evaluate_method, model.save_method, [self.error])
            classifier.split_data()
            classifier.set_num_folds()
            classifier.train()
            classifier.evaluate(save = save, save_dir = save_dir)
            return classifier.get_test_metric(self.error)
        return inner_eval

    def run_optimization(self, save_dir):
        self.result = gp_minimize(self.single_iter(self.X, self.y), parameters)
        save_optimization(save_dir)

    def save_optimization(self, save_dir):
        np.save(os.path.join(save_dir, "sol.npy"), self.result.x)
        self.single_iter(self.X, self.y, save = True, save_dir = save_dir)
