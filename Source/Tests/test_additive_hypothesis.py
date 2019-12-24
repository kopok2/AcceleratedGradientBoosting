# coding=utf-8
"""Example test for build."""

import unittest
import AcceleratedGradientBoosting.Source.AcceleratedGradientBoosting as AGB


class TestSuite(unittest.TestCase):
    def test_adding(self):
        agb = AGB.AcceleratedGradientBoosting()

if __name__ == '__main__':
    unittest.main()
