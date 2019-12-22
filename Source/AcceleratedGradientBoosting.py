# coding=utf-8
"""
Module implements State-of-the-Art classification and regression machine learning algorithm -
   Accelerated Gradient Boosting.
   
   Research source:
   link: https://arxiv.org/pdf/1803.02042.pdf
   
   Abstract:
   Gradient tree boosting is a prediction algorithm that sequentially produces a model in the form of
   linear combinations of decision trees, by solving an infinite-dimensional optimization problem.
   We combine gradient boosting and Nesterov’s accelerated descent to design a new algorithm,
   which we callAGB(for Accelerated Gradient Boosting).
   Substantial numerical evidence is provided on both synthetic and real-life data sets
   to assess the excellent performance of the method in a large variety of prediction problems.
   It is empirically shown that AGBis much less sensitive to the shrinkage parameter
   and outputs predictors that are considerably more sparse in the number of trees,
   while retaining the exceptional performance of gradient boosting.

Implementation by Karol Oleszek 2019
"""
from functools import reduce
from operator import add
import numpy as np
from sklearn.tree import DecisionTreeRegressor


class AdditiveHypothesis:
    """
    Additive Hypothesis class for linear combination of functions.
    """
    def __init__(self):
        self.predictors = []
        self.weights = []
        self.predict_cache = None

    def infer(self, X, cache=False):
        """
        Get predictions from additive model.
        
        Supports training time caching.
        """
        result = None
        if self.predict_cache:
            result = self.predict_cache + self.predictors[-1].predict(X) * self.weights[-1]
        else:
            result = np.zeros(len(X))
            for predictor, weight in zip(self.predictors, self.weights):
                result += predictor.predict(X) * weight
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


class AcceleratedGradientBoosting:
    """
    Class implements additive function fitting using base-learner agnostic learning procedure.
    """
    def __init__(self, iterations, base_learner=DecisionTreeRegressor, 
                 base_learner_params={'max_depth': 5}, shrinkage=0.9):
        self.iterations = iterations
        self.base_learner = base_learner
        self.base_learner_params = base_learner_params

    def fit(X, y):
        """
        Perform additive learning procedure.
        """
        