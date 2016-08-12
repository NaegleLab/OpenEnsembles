"""
OpenEnsembles is a resource for performing and analyzing ensemble clustering
This file contains all transform functions. Each transform takes a data matrix
an x-vector and variable arguments. It must return a data matrix, an x-vector
and a dictionary of parameters used (name, value)
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

class transforms:
    def __init__(self, x, data, kwargs):
        """
        transforms object intiatilizes the object with .x, .data, and .args. The object is modified after a transformation
        to report the .x_out, .data_out and .var_params
        """
        self.x = x
        self.data = data
        self.args = kwargs
        self.x_out = []
        self.data_out = []
        self.var_params = {}

    def transforms_available(self):
        methods =  [method for method in dir(self) if callable(getattr(self, method))]
        methods.remove('__init__')
        methods.remove('transforms_available')
        methodDict = {}
        for method in methods:
            methodDict[method] = ''
        return methodDict


    def zscore(self):
        self.x_out = self.x 
        self.data_out = stats.zscore(self.data, 1)
        self.var_params = {}


    def minmax(self):
        if 'minValue' in self.args:
            minValue = self.args['minValue']
        else:
            minValue = 0
        if 'maxValue' in self.args:
            maxValue = self.args['maxValue']
        else:
            maxValue = 1

        if minValue > maxValue:
            raise ValueError("Your requested minValue (%0.2f) is larger than the maximum value (%0.2f)"%(minValue, maxValue))

        self.x_out = self.x
        min_max_scaler = preprocessing.MinMaxScaler(feature_range=(minValue, maxValue))
        self.data_out = np.transpose(min_max_scaler.fit_transform(np.transpose(self.data)))
        self.var_params = {'minValue':minValue, 'maxValue':maxValue}
        
    def log(self):
        """
        Log transformation will default to taking the log2 of all elements in the matrix. 
        Use base=2, base=10, base='e' or base='ln'
        It is a good idea to check for the presence of infinite values created
        """
        if 'base' in self.args:
            base = self.args['base']
        else:
            base = 2 
        self.var_params['base'] = base
        self.x_out = self.x
        if isinstance(base, basestring):
            if base == 'e' or base=='ln':
                self.data_out = np.log(self.data)
        elif int(base) == 10:
            self.data_out = np.log10(self.data)
        elif int(base) == 2:
            self.data_out = np.log2(self.data)
        else:
            raise ValueError('Requested base for logarithm was not recognized as either e, 2, or 10)')




        

