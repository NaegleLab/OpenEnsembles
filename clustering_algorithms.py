"""
OpenEnsembles is a resource for performing and analyzing ensemble clustering
This file contains calls to clustering algorithms
"""

import numpy as np 
import pandas as pd 
import sklearn.cluster as skc
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy.cluster.hierarchy as sch
from sklearn import preprocessing
import scipy.stats as stats
from types import FunctionType

class clustering_algorithms:
    def __init__(self, data, kwargs):
        """
        Clustering objects are initialized with the data they act on and the variable arguments
        The act of clustering creates a list of assignments of objects in data assigned to clustering classes
        var_params contains all the final parameters used in the act of clustering
        """
        self.data = data
        self.args = kwargs
        self.out = []
        self.var_params = {}
        #args should have K, even if a default value
        if 'K' not in self.args:
            raise ValueError('clustering_algorithms should have an instantiated K as part of kwargs key, pair')

    def clustering_algorithms_available(self):
        methods =  [method for method in dir(self) if callable(getattr(self, method))]
        methods.remove('__init__')
        methods.remove('clustering_algorithms_available')
        methodDict = {}
        for method in methods:
            methodDict[method] = ''
        return methodDict

    def kmeans(self):
        pass


