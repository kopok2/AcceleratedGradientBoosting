# coding=utf-8
"""Test accelerated gradient boosting."""

import os
import sys
import unittest
import numpy as np
from sklearn import datasets, model_selection, metrics, tree

TEST_DIR = os.path.dirname(__file__)
SRC_DIR = '../Source'
sys.path.insert(0, os.path.abspath(os.path.join(TEST_DIR, SRC_DIR)))

import AcceleratedGradientBoosting

TEST_DATA_LEN = 100_000


class DummyPredictor:
    """
    Dummy predictor for test purposes.
    """
    def __init__(self, value):
        self.value = value

    def predict(self, data_x):
        """
        Get constant value.

        :param data_x: numpy array.
        :return: constant numpy array.
        """
        return np.repeat(self.value, len(data_x))


class TestSuite(unittest.TestCase):
    def test_creation(self):
        agb = AcceleratedGradientBoosting.AcceleratedGradientBoosting()
        self.assertEqual(200, agb.iterations)
        self.assertEqual(tree.DecisionTreeRegressor, agb.base_learner)
        self.assertEqual(None, agb.base_learner_params)
        self.assertEqual(0.9, agb.shrinkage)
        self.assertEqual(None, agb.hypothesis)
        self.assertEqual(2, agb.n_classes)

    def test_agb_fit_predict(self):
        x_data, y_data = datasets.load_iris(return_X_y=True)
        x_train, x_test, y_train, y_test = model_selection.train_test_split(x_data, y_data)

        agb = AcceleratedGradientBoosting.AcceleratedGradientBoosting(iterations=100, n_classes=3,
                                                                      base_learner_params={'max_leaf_nodes': 10})
        agb.fit(x_train, y_train)

        self.assertGreater(metrics.accuracy_score(y_test, agb.predict(x_test)), 0.9)
        self.assertGreater(metrics.accuracy_score(y_train, agb.predict(x_train)), 0.95)


if __name__ == '__main__':
    unittest.main()
