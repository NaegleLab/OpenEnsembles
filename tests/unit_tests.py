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
        self.assertRaises(ValueError, lambda: oe.data(df,x)) 

    def test_transform_error_noSource(self):
        #self.assertEqual(-1, self.data.transform('parentZ', 'zscore', 'zscore_parentZ'))
        self.assertRaises(ValueError, lambda: self.data.transform('parentZ', 'zscore', 'zscore_parentZ'))

    def test_transform_NoNaNs(self):
        self.data.transform('parent', 'zscore', 'zscore', Keep_NaN=0)
        self.assertEqual(1, len(self.data.D))

    def test_transform_NoInf(self):
        self.data.transform('parent', 'log', 'log10', base=10, Keep_Inf=0)
        self.assertEqual(1, len(self.data.D))

    def test_transform_error_noTxfm(self):
        self.assertRaises(ValueError, lambda: self.data.transform('parent', 'zscoreGobblyGook', 'zscore'))

    def test_all_transform(self):
        #check to see that a new entry in D, x and params are added for every
        #transform available
        TXFM_FCN_DICT = self.data.transforms_available()
        len_expected = 2
        for transform in TXFM_FCN_DICT:
            self.data.transform('parent', transform, transform)
            self.assertEqual(len_expected, len(self.data.D))
            self.assertEqual(len_expected, len(self.data.x))
            self.assertEqual(len_expected, len(self.data.params))
            len_expected += 1

    def test_clustering_setup(self):
        c = oe.cluster(self.data)
        self.assertEqual(1, len(c.dataObj.D))

    def test_clustering_NoSource(self):
        c = oe.cluster(self.data)
        self.assertRaises(ValueError, lambda: c.cluster('parentZ', 'kmeans', 'bad'))

    def test_clustering_NoAlgorithm(self):
        c = oe.cluster(self.data)
        self.assertRaises(ValueError, lambda: c.cluster('parent', 'gobblygook', 'bad'))

    def test_clustering_namingTestRequireUnique(self):
        c = oe.cluster(self.data)
        c.cluster('parent', 'kmeans', 'kmeans', K=2)
        self.assertRaises(ValueError, lambda: c.cluster('parent', 'kmeans', 'kmeans', Require_Unique=1, K=2))

    def test_clustering_namingTestRequireNotUnique(self):
        c = oe.cluster(self.data)
        c.cluster('parent', 'kmeans', 'kmeans', K=2)
        c.cluster('parent', 'kmeans', 'kmeans', Require_Unique=0, K=2)
        self.assertEqual(2, len(c.labels))

    def test_validation_badSourceAndCluster(self):
        c = oe.cluster(self.data)
        c.cluster('parent', 'kmeans', 'kmeans', K=2)
        v = oe.validation(self.data, c)
        self.assertRaises(ValueError, lambda: v.calculate('Ball_Hall', 'kmeans', 'gobblygook'))
        self.assertRaises(ValueError, lambda: v.calculate('Ball_Hall', 'gobblygook', 'parent'))
        self.assertRaises(ValueError, lambda: v.calculate('GobblyGook', 'kmeans', 'parent'))

    def test_ReplicateValidation(self):
        c = oe.cluster(self.data)
        c.cluster('parent', 'kmeans', 'kmeans', K=2)
        v = oe.validation(self.data, c)
        len_expected = 0
        self.assertEqual(len_expected, len(v.validation))
        v.calculate('Ball_Hall', 'kmeans', 'parent')
        len_expected = 1
        self.assertEqual(len_expected, len(v.validation))
        v.calculate('Ball_Hall', 'kmeans', 'parent')
        self.assertEqual(len_expected, len(v.validation))



    def test_allValidationMetrics(self):
        c = oe.cluster(self.data)
        c.cluster('parent', 'kmeans', 'kmeans', K=2)
        v = oe.validation(self.data, c)
        FCN_DICT = v.validation_metrics_available()
        len_expected = 1
        for validation_name in FCN_DICT:
            v.calculate(validation_name, 'kmeans', 'parent')
            self.assertEqual(len_expected, len(v.validation))
            self.assertEqual(len_expected, len(v.description))
            self.assertEqual(len_expected, len(v.source_name))
            self.assertEqual(len_expected, len(v.cluster_name))
            len_expected += 1





