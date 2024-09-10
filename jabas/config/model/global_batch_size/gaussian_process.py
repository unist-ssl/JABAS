import os
import numpy as np
import pickle

from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import *
from sklearn.utils._testing import ignore_warnings
from sklearn.exceptions import ConvergenceWarning


class GaussianProcessRegressionModel(object):
    def __init__(self):
        self.kernel = ConstantKernel(constant_value=1,constant_value_bounds =(1,1e6)) * \
                        ExpSineSquared(1.0, 3, periodicity_bounds=(1e-2, 10))
        self.gpr = GaussianProcessRegressor(self.kernel, n_restarts_optimizer=9, normalize_y=True)

    @ignore_warnings(category=ConvergenceWarning)
    def train(self, x_train_list, y_train_list):
        if x_train_list is None or y_train_list is None:
            raise ValueError(f"Argument is None - x_train_list: {x_train_list} | y_train_list: {y_train_list}")
        else:
            if not isinstance(x_train_list, (list, tuple)):
                raise ValueError(
                    f"Argument x_train_list must be list or tuple type: {type(x_train_list)}")
            if not isinstance(y_train_list, (list, tuple)):
                raise ValueError(
                    f"Argument y_train_list must be list or tuple type: {type(y_train_list)}")
        if len(x_train_list) != len(y_train_list):
            raise ValueError('Argument x_train_list and y_train_list must have equal length, '
                             f'but {len(x_train_list)} and {len(y_train_list)}')

        x_train = np.array(x_train_list).reshape(-1, 1)
        y_train = np.array(y_train_list)
        self.gpr.fit(x_train, y_train)

    def evaluate(self, x_pred_list):
        x_pred = np.array(x_pred_list).reshape(-1, 1)
        y_pred_mean, y_pred_std = self.gpr.predict(x_pred, return_std=True)
        return y_pred_mean

    def save(self, checkpoint_dir):
        ckpt_file_path = os.path.join(checkpoint_dir, 'gaussian_process_model.sav')
        pickle.dump(self.gpr, open(ckpt_file_path, 'wb'))

    def load(self, checkpoint_dir):
        ckpt_file_path = os.path.join(checkpoint_dir, 'gaussian_process_model.sav')
        self.gpr = pickle.load(open(ckpt_file_path, 'rb'))