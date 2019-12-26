# coding=utf-8
"""
Symbolic algebraic hypothesis module.
"""
from collections import defaultdict
import numpy as np


class AlgebraicHypothesis:
    """
    Symbolic algebraic hypothesis supporting adding and multiplication.
    """
    def __init__(self):
        self.symbol_weights = defaultdict(int)
        self.symbol_evaluation_scheme = {}
        self.train_cache = {}

    def evaluate(self, data_x, cache=False):
        """
        Evaluate hypothesis against input data.

        :param data_x: numpy array sized accordingly to evaluation scheme members.
        :param cache: training time cache indicator.
        :return: numpy array.
        """
        result = np.zeros(len(data_x))
        for symbol, weight in self.symbol_weights.items():
            if symbol in self.train_cache:
                result += weight * self.train_cache[symbol]
            else:
                eval_result = self.symbol_evaluation_scheme[symbol].predict(data_x)
                if cache:
                    self.train_cache[symbol] = eval_result
                result += weight * eval_result
        return result

    def add_symbol(self, symbol, weight, evaluation_scheme):
        """
        Add symbol to algebraic expression.

        :param symbol: identifying symbol (str).
        :param weight: symbol weight (float).
        :param evaluation_scheme: numpy predictor.
        """
        self.symbol_weights[symbol] = weight
        self.symbol_evaluation_scheme[symbol] = evaluation_scheme

    def __add__(self, other):
        new_expression = AlgebraicHypothesis()
        for symbol, weight in self.symbol_weights.items():
            new_expression.symbol_weights[symbol] += weight
            new_expression.symbol_evaluation_scheme[symbol] = self.symbol_evaluation_scheme[symbol]
        for symbol, weight in other.symbol_weights.items():
            new_expression.symbol_weights[symbol] += weight
            new_expression.symbol_evaluation_scheme[symbol] = other.symbol_evaluation_scheme[symbol]
        return new_expression

    def __mul__(self, other):
        for symbol in self.symbol_weights.keys():
            self.symbol_weights[symbol] *= other
        return self

    def __str__(self):
        result = ["Algebraic hypothesis:"]
        for symbol, weight in self.symbol_weights.items():
            result.append(f"{symbol} * {weight} | {self.symbol_evaluation_scheme[symbol]}")
        return "\n".join(result)

    def clear_cache(self):
        """
        Clear train time result cache.
        """
        self.train_cache = {}
