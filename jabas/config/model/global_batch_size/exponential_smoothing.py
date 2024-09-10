import os
import numpy as np
import pickle

from statsmodels.tsa.holtwinters import ExponentialSmoothing as ETS
from statsmodels.iolib.smpickle import load_pickle


class ExponentialSmoothing(object):
    def __init__(self):
        self.ets_fit = None
        self.total_y = []

    def train(self, x_train_list, y_train_list):
        self.total_y.extend(y_train_list)
        y_train = np.array(self.total_y)
        try:
            self.ets_fit = ETS(y_train, seasonal=None, seasonal_periods=None).fit()
        except Exception as e:
            print(f'[ERROR][{self.__class__.__name__}] y_train: {y_train} | '
                  f'Training ExponentialSmoothing requires at least two data samples, '
                  f'but {len(y_train)}')
            raise e

    def evaluate(self, x_pred_list):
        assert self.ets_fit is not None
        x_pred = (np.array(x_pred_list) / 100).astype(int)
        start_x = int(x_pred[0])
        end_x = int(x_pred[-1])
        last_index = len(self.ets_fit.fittedvalues) - 1
        start_x = start_x - last_index - 1
        end_x = end_x - last_index
        y_pred_mean = self.ets_fit.forecast(steps=end_x)[start_x:]
        return y_pred_mean

    def save(self, checkpoint_dir):
        ckpt_file_path = os.path.join(checkpoint_dir, 'exp_smoothing_model.pkl')
        total_y_ckpt_file_path = os.path.join(checkpoint_dir, 'exp_smoothing_total_y')
        if self.ets_fit:
            self.ets_fit.save(ckpt_file_path)
        if self.total_y:
            with open(total_y_ckpt_file_path, 'wb') as f:
                pickle.dump(self.total_y, f)

    def load(self, checkpoint_dir):
        ckpt_file_path = os.path.join(checkpoint_dir, 'exp_smoothing_model.pkl')
        total_y_ckpt_file_path = os.path.join(checkpoint_dir, 'exp_smoothing_total_y')
        if os.path.isfile(ckpt_file_path):
            self.ets_fit = load_pickle(ckpt_file_path)
            with open(total_y_ckpt_file_path, 'rb') as f:
                self.total_y = pickle.load(f)