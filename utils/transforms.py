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

def zscore(x, data, **kwargs):
    X = x 
    DATA = stats.zscore(data, 1)
    params = {}
    return (X, DATA, params)

def minmax(x, data, minValue=0, maxValue=1, **kwargs):
    if 'minValue' in kwargs:
        minValue = kwargs['minValue']
    if 'maxValue' in kwargs:
        maxValue = kwargs['maxValue']

    if minValue > maxValue:
        print "ERROR: Your requested minValue (%0.2f) is larger than the maximum value (%0.2f)"%(minValue, maxValue)
        return -2
    X = x
    min_max_scaler = preprocessing.MinMaxScaler(feature_range=(minValue, maxValue))
    DATA = np.transpose(min_max_scaler.fit_transform(np.transpose(data)))
    return (X, DATA, {'minValue':minValue, 'maxValue':maxValue})
