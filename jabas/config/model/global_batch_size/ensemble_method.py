import numpy as np


class EnsembleMethod(object):
    def __init__(self, models=[], rates=[]):
        self.models = models
        self.rates = rates
        if len(self.models) == 0 or len(self.rates) == 0:
            raise ValueError(
                f'len(self.models) and len(self.rates) must > 0, '
                f'but {len(self.models)} and {len(self.rates)}')
        if len(self.models) != len(self.rates):
            raise ValueError(
                f'len(self.models) must be equal to len(self.rates), '
                f'but {len(self.models)} and {len(self.rates)}')
        if sum(rates) != 1:
            raise ValueError(
                f'sum value of rates must be 1, but {sum(rates)}'
            )

    def train(self, x_train_list, y_train_list):
        for model in self.models:
            model.train(x_train_list, y_train_list)

    def evaluate(self, x_pred_list):
        ensemble_y_pred = None
        for i, (model, rate) in enumerate(zip(self.models, self.rates)):
            y_pred = model.evaluate(x_pred_list)
            if i == 0:
                ensemble_y_pred = np.zeros_like(y_pred)
            ensemble_y_pred += (y_pred * rate)
        return ensemble_y_pred

    def save(self, checkpoint_dir):
        for model in self.models:
            model.save(checkpoint_dir)

    def load(self, checkpoint_dir):
        for model in self.models:
            model.load(checkpoint_dir)