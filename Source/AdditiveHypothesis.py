# coding=utf-8
"""
Additive Hypothesis Module.
"""
import numpy as np


class AdditiveHypothesis:
    """
    Additive Hypothesis class for linear combination of functions.
    """

    def __init__(self):
        self.predictors = []
        self.weights = []
        self.predict_cache = None

    def infer(self, data_x, cache=False):
        """
        Get predictions from additive model.

        Supports training time caching.
        """
        if self.predict_cache:
            result = self.predict_cache + self.predictors[-1].predict(data_x) * self.weights[-1]
        else:
            result = np.zeros(len(data_x))
            for predictor, weight in zip(self.predictors, self.weights):
                result += predictor.predict(data_x) * weight
        if cache:
            self.predict_cache = result
        return result

    def add_predictor(self, predictor, weight):
        """
        Expand additive model with next predictor and respective weight.
        """
        self.predictors.append(predictor)
        self.weights.append(weight)

    def clear_cache(self):
        """
        Clear learning time cache.
        """
        self.predict_cache = None
