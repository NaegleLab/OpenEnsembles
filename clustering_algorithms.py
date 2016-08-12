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
    def __init__(self, data, kwargs, K=2):
        """
        Clustering objects are initialized with the data they act on and the variable arguments
        The act of clustering creates a list of assignments of objects in data assigned to clustering classes
        var_params contains all the final parameters used in the act of clustering
        """
        self.data = data
        self.out = []
        self.var_params = kwargs
        self.K = K
        #args should have K, even if a default value
        #if 'K' not in self.args:
        #    raise ValueError('clustering_algorithms should have an instantiated K as part of kwargs key, pair')

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

    def spectral(self):
        """ 
        Calls skc.SpectralClustering()
                solution = skc.SpectralClustering(n_clusters=self.K, n_components=params['n_components'],
                        eigen_solver=params['eigen_solver'], random_state=['random_state'], n_init=params['n_init'],
                        affinity=params['affinity'], 
                        eigen_tol=params['eigen_tol'], assign_labels=params['assign_labels'])
        """
        params = {}

        params['eigen_solver']=None
        params['random_state'] = None
        params['n_init'] = 10
        params['affinity'] = 'rbf'
        params['n_neighbors'] = 10
        params['eigen_tol'] = '0.0'
        params['assign_labels'] = 'kmeans'

        #for anything in self.var_params that may replace defaults, update the param list
        overlap = set(params.keys()) & set(self.var_params.keys())
        for key in overlap:
            params[key] = self.var_params[key]

 
        solution = skc.SpectralClustering(n_clusters=self.K, n_neighbors=params['n_neighbors'],
                        eigen_solver=params['eigen_solver'], random_state=params['random_state'], n_init=params['n_init'],
                        affinity=params['affinity'], 
                        eigen_tol=params['eigen_tol'], assign_labels=params['assign_labels'])
        solution.fit(self.data)
        self.out = solution.labels_
        self.var_params = params #update dictionary of parameters to match that used.
       


