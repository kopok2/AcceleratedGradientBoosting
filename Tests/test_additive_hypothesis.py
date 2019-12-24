# coding=utf-8
"""Example test for build."""

import os
import sys

testdir = os.path.dirname(__file__)
srcdir = '../Source'
sys.path.insert(0, os.path.abspath(os.path.join(testdir, srcdir)))
import unittest
import AcceleratedGradientBoosting as AGB


class TestSuite(unittest.TestCase):
    def test_adding(self):
        agb = AGB.AcceleratedGradientBoosting()

if __name__ == '__main__':
    unittest.main()
