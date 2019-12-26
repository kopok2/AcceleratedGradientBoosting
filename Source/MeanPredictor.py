# coding=utf-8
"""
Simple mean predictor module.

Used for initial fitting of additive hypothesis to minimize mean squared error in first iteration.
"""
import numpy as np


class MeanPredictor:
    """
    Mean predictor class, which returns precomputed mean constant.
    """
    def __init__(self):
        self.mean = 0

    def fit(self, y):
        """
        Fit mean.

        :param y: numpy array.
        """
        self.mean = np.mean(y)

    def predict(self, data_x):
        """
        Get repeated constant prediction.

        :param data_x: numpy array.
        :return: numpy array.
        """
        return np.repeat(self.mean, len(data_x))
