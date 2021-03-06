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
import math

import numpy as np
from sklearn.tree import DecisionTreeRegressor
from MeanPredictor import MeanPredictor
from AlgebraicHypothesis import AlgebraicHypothesis


class AcceleratedGradientBoosting:
    """
    Class implements additive function fitting using base-learner agnostic learning procedure.
    """
    def __init__(self, iterations=200, base_learner=DecisionTreeRegressor,
                 base_learner_params=None, shrinkage=0.9, n_classes=2):
        self.iterations = iterations
        self.base_learner = base_learner
        self.base_learner_params = base_learner_params
        self.shrinkage = shrinkage
        self.hypothesis = None
        self.n_classes = n_classes

    def fit(self, data_x, y, verbose=False):
        """
        Perform additive learning procedure.
        """
        # Initialize Nesterov learning scheme
        lambda_now = 0
        gamma = 1

        # Initialize hypothesis
        f_now = AlgebraicHypothesis()
        mp = MeanPredictor()
        mp.fit(y)
        f_now.add_symbol("Mean_start", 1.0, mp)
        g_now = AlgebraicHypothesis()
        g_now.add_symbol("Mean_start", 1.0, mp)

        # Perform boosting
        for epoch in range(self.iterations):
            if verbose:
                print(f"Epoch: {epoch + 1}")
            g_prev = g_now
            # Compute gradient
            gradient = y - g_prev.evaluate(data_x, True)
            # Fit base learner to gradient
            if self.base_learner_params:
                base_learner_part = self.base_learner(**self.base_learner_params)
            else:
                base_learner_part = self.base_learner()
            base_learner_part.fit(data_x, gradient)
            base_learner_name = f"BaseLearner {epoch}"

            # Update
            f_prev = f_now
            f_now = AlgebraicHypothesis()
            f_now.add_symbol(base_learner_name, self.shrinkage, base_learner_part)
            f_now = g_prev + f_now
            g_now = (f_now * (1 - gamma)) + (f_prev * gamma)

            # Update Nesterov scheme
            lambda_prev = lambda_now
            lambda_now = (1 + math.sqrt(1 + 4 * lambda_prev)) / 2
            gamma = (1 - lambda_prev) / lambda_now
        f_now.clear_cache()
        self.hypothesis = f_now

    def predict(self, data_x):
        """
        Get hypothesis predictions.

        :param data_x: numpy array.
        :return: prediction vector.
        """
        bins = [x + 0.5 for x in range(self.n_classes - 1)]
        return np.digitize(self.hypothesis.evaluate(data_x), bins)
