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
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from MeanPredictor import MeanPredictor


class AcceleratedGradientBoosting:
    """
    Class implements additive function fitting using base-learner agnostic learning procedure.
    """
    def __init__(self, iterations=1000, base_learner=DecisionTreeRegressor,
                 base_learner_params=None, shrinkage=0.9):
        self.iterations = iterations
        self.base_learner = base_learner
        self.base_learner_params = base_learner_params
        self.shrinkage = shrinkage

    def fit(self, data_x, y):
        """
        Perform additive learning procedure.
        """
        # Initialize Nesterov learning scheme
        lambda_prev = None
        lambda_now = 0
        gamma = 1

        # Initialize hypothesis
