# coding=utf-8
"""Test Mean predictor."""

import os
import sys
import unittest
import numpy as np

TEST_DIR = os.path.dirname(__file__)
SRC_DIR = '../Source'
sys.path.insert(0, os.path.abspath(os.path.join(TEST_DIR, SRC_DIR)))

import MeanPredictor

TEST_DATA_LEN = 100_000


class TestSuite(unittest.TestCase):
    def test_creation(self):
        mean_p = MeanPredictor.MeanPredictor()
        self.assertEqual(0, mean_p.mean)

    def test_fit(self):
        mean_p = MeanPredictor.MeanPredictor()
        data = np.zeros(TEST_DATA_LEN)
        mean_p.fit(data)
        self.assertAlmostEqual(0, mean_p.mean)

    def test_predict(self):
        mean_p = MeanPredictor.MeanPredictor()
        data = np.ones(TEST_DATA_LEN)
        mean_p.fit(data)
        data_width = 20
        self.assertTrue(np.allclose(1, mean_p.predict(np.random.random((TEST_DATA_LEN, data_width)))))


if __name__ == '__main__':
    unittest.main()
