# coding=utf-8
"""Test algebraic hypothesis."""

import os
import sys
import unittest
import numpy as np

TEST_DIR = os.path.dirname(__file__)
SRC_DIR = '../Source'
sys.path.insert(0, os.path.abspath(os.path.join(TEST_DIR, SRC_DIR)))

import AlgebraicHypothesis

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
        algebraic_h = AlgebraicHypothesis.AlgebraicHypothesis()
        self.assertEqual({}, algebraic_h.symbol_weights)
        self.assertEqual({}, algebraic_h.symbol_evaluation_scheme)

    def test_hypothesis_symbol_adding(self):
        algebraic_h = AlgebraicHypothesis.AlgebraicHypothesis()
        algebraic_h.add_symbol("WeakLearner_1", 0.12, None)
        self.assertEqual(0.12, algebraic_h.symbol_weights["WeakLearner_1"])
        self.assertEqual(None, algebraic_h.symbol_evaluation_scheme["WeakLearner_1"])

    def test_hypothesis_evaluation(self):
        algebraic_h = AlgebraicHypothesis.AlgebraicHypothesis()
        algebraic_h.add_symbol("WeakLearner_1", 0.12, DummyPredictor(100))
        algebraic_h.add_symbol("WeakLearner_2", 1, DummyPredictor(13))
        data_width = 100
        self.assertTrue(np.allclose(np.repeat(25, TEST_DATA_LEN),
                                    algebraic_h.evaluate(np.random.random((TEST_DATA_LEN, data_width)))))

    def test_hypothesis_adding(self):
        algebraic_h_1 = AlgebraicHypothesis.AlgebraicHypothesis()
        algebraic_h_1.add_symbol("WeakLearner_1", 0.12, DummyPredictor(100))
        algebraic_h_2 = AlgebraicHypothesis.AlgebraicHypothesis()
        algebraic_h_2.add_symbol("WeakLearner_2", 1, DummyPredictor(13))
        algebraic_h_3 = algebraic_h_1 + algebraic_h_2
        data_width = 100
        self.assertTrue(np.allclose(np.repeat(25, TEST_DATA_LEN),
                                    algebraic_h_3.evaluate(np.random.random((TEST_DATA_LEN, data_width)))))

    def test_hypothesis_multiplication(self):
        algebraic_h = AlgebraicHypothesis.AlgebraicHypothesis()
        algebraic_h.add_symbol("WeakLearner_1", 0.12, DummyPredictor(100))
        algebraic_h.add_symbol("WeakLearner_2", 1, DummyPredictor(13))
        algebraic_h *= 10
        data_width = 100
        self.assertTrue(np.allclose(np.repeat(250, TEST_DATA_LEN),
                                    algebraic_h.evaluate(np.random.random((TEST_DATA_LEN, data_width)))))

    def test_hypothesis_printing(self):
        algebraic_h = AlgebraicHypothesis.AlgebraicHypothesis()
        algebraic_h.add_symbol("WeakLearner_1", 0.12, None)
        self.assertEqual("Algebraic hypothesis:\nWeakLearner_1 * 0.12 | None", str(algebraic_h))


if __name__ == '__main__':
    unittest.main()
