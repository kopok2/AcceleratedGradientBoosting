# coding=utf-8
"""Test Additive Hypothesis."""

import os
import sys
import unittest
import numpy as np

TEST_DIR = os.path.dirname(__file__)
SRC_DIR = '../Source'
sys.path.insert(0, os.path.abspath(os.path.join(TEST_DIR, SRC_DIR)))

import AdditiveHypothesis

TEST_DATA_LEN = 100_000


class DummyPredictor:
    """
    Dummy predictor for testing.
    """
    def __init__(self, val):
        self.val = val

    def predict(self, data_x):
        """
        Get dummy prediction of predefined value.

        :param data_x: numpy array.
        :return: dummy value repeated respectively.
        """
        return np.repeat(self.val, len(data_x))


class TestSuite(unittest.TestCase):
    def test_creation(self):
        add_hypothesis = AdditiveHypothesis.AdditiveHypothesis()
        self.assertEqual([], add_hypothesis.predictors)
        self.assertEqual([], add_hypothesis.weights)
        self.assertEqual(None, add_hypothesis.predict_cache)

    def test_basic_inference(self):
        add_hypothesis = AdditiveHypothesis.AdditiveHypothesis()
        add_hypothesis.add_predictor(DummyPredictor(-1), 0.5)
        add_hypothesis.add_predictor(DummyPredictor(1), 0.5)
        self.assertTrue(all(np.zeros(TEST_DATA_LEN) == add_hypothesis.infer(np.random.random((TEST_DATA_LEN, 10)))))

    def test_cached_inference(self):
        add_hypothesis = AdditiveHypothesis.AdditiveHypothesis()
        add_hypothesis.add_predictor(DummyPredictor(-1), 0.5)
        self.assertTrue(all(np.repeat(-0.5, TEST_DATA_LEN) ==
                            add_hypothesis.infer(np.random.random((TEST_DATA_LEN, 10)), True)))
        add_hypothesis.add_predictor(DummyPredictor(1), 0.5)
        self.assertTrue(all(np.zeros(TEST_DATA_LEN) ==
                            add_hypothesis.infer(np.random.random((TEST_DATA_LEN, 10)), True)))

    def test_cache_clear(self):
        add_hypothesis = AdditiveHypothesis.AdditiveHypothesis()
        add_hypothesis.add_predictor(DummyPredictor(-1), 0.5)
        add_hypothesis.infer(np.random.random((TEST_DATA_LEN, 10)), True)
        add_hypothesis.add_predictor(DummyPredictor(1), 0.5)
        add_hypothesis.infer(np.random.random((TEST_DATA_LEN, 10)), True)
        add_hypothesis.clear_cache()
        self.assertEqual(None, add_hypothesis.predict_cache)

    def test_load_caching(self):
        load_rounds = 100
        add_hypothesis = AdditiveHypothesis.AdditiveHypothesis()
        for i in range(load_rounds):
            add_hypothesis.add_predictor(DummyPredictor(1), 1 / load_rounds)
            self.assertTrue(np.allclose(np.repeat((i + 1) / load_rounds, TEST_DATA_LEN),
                                        add_hypothesis.infer(np.random.random((TEST_DATA_LEN, 10)), True)))
        add_hypothesis.clear_cache()
        self.assertTrue(np.allclose(np.ones(TEST_DATA_LEN),
                                    add_hypothesis.infer(np.random.random((TEST_DATA_LEN, 10)))))


if __name__ == '__main__':
    unittest.main()
