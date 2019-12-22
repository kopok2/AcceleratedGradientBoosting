# coding=utf-8
"""Example test for build."""

import unittest


class TestSuite(unittest.TestCase):
	def test_adding(self):
		self.assertEqual(2 + 2, 4)


if __name__ == '__main__':
	unittest.main()
