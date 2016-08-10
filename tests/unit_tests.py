import os.path
import time
import unittest
import random
import pandas as pd

import openensembles as oe

class TestFunctions(unittest.TestCase):

    def setUp(self):
        fileName = 'data_test.csv'
        df = pd.DataFrame.from_csv(fileName)
        x = [0, 5, 30]
        self.data = oe.data(df, x)

    def test_correct_setup(self):
        self.assertEqual((3,3), self.data.D['parent'].shape)
        self.assertEqual(3, len(self.data.x['parent']))

    def test_incorrect_setup(self):
        fileName = 'data_test.csv'
        df = pd.DataFrame.from_csv(fileName)
        x = [0, 5, 10, 30]
        data = oe.data(df, x)
        self.assertEqual((3,3), data.D['parent'].shape)
        self.assertEqual(3, len(data.x['parent']))

    def test_transform_error_noSource(self):
        self.assertEqual(-1, self.data.transform('parentZ', 'zscore', 'zscore_parentZ', []))

    def test_transform_error_noTxfm(self):
        self.assertEqual(-2, self.data.transform('parent', 'zscoreGobblyGoodk', 'zscore_parent', []))

