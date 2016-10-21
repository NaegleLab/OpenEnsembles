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
import collections
import re
from sklearn.decomposition import PCA

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
        methods =  [method for method in dir(self) if isinstance(getattr(self, method), collections.Callable)]
        methods.remove('transforms_available')
        methodDict = {}
        for method in methods:
            if not re.match('__', method):
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
        if isinstance(base, str):
            if base == 'e' or base=='ln':
                self.data_out = np.log(self.data)
        elif int(base) == 10:
            self.data_out = np.log10(self.data)
        elif int(base) == 2:
            self.data_out = np.log2(self.data)
        else:
            raise ValueError('Requested base for logarithm was not recognized as either e, 2, or 10)')


    def PCA(self):
        """
        Applies PCA to data matrix. If variable argument n_components is set, it will keep the first n_components of the 
        post-transformed data.
        set n_components to a number between 0 and 1 to reduce dimensionality based on %variance explained.
        """

        if 'n_components' in self.args:
            n_components = self.args['n_components']
        else:
            n_components = self.data.shape[1]
        self.var_params['n_components'] = n_components
        pca = PCA(n_components=n_components)
        pca.fit(self.data)
        self.data_out = pca.transform(self.data)
        self.x_out = []
        for i in range(0, self.data_out.shape[1]):
            self.x_out.append("PC%d"%(i+1))
            
        return pca

    def internal_normalization(self):
        """
        This normalizes all data (in rows) to one point in that row, based on either
        col_index or value in the x vector (x_val)
        Example:
        Normalize data to 5minutes internal_normalization(x_val=5)
        """
        if 'col_index' in self.args:
            index = self.args['col_index']
            if index >= len(self.x):
                raise ValueError('internal_normalization requires an index value within the length of the x vector')
            x_val = self.x[index]

        elif 'x_val' in self.args:
            #find col_name
            x_val = self.args['x_val']
            indexes = np.flatnonzero(np.asarray(self.x) == x_val)
            if not indexes:
                raise ValueError('internal_normalization requires an x_val in the x vector, %s was not found'%(x_val))
            elif len(indexes) > 1:
                raise ValueError('internal_normalization requires an x_val in the x vector that appears one time, %s was found %d tiems'%(x_val, len(indexes)))
            else:
                index = indexes[0]
        else:
            raise ValueError('internal_normalization requires a valid normalizing column according to a specific value in x (x_val) or index (col_index) ')

        self.var_params['x_val'] = x_val
        self.var_params['index'] = index
        self.x_out = self.x
        vec = self.data[:,index]
        self.data_out = self.data/vec[:,None]




        

