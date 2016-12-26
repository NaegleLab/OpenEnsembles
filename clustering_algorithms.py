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
import re
import warnings

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
        """
        self.clustering_algorithms_available() returns a dictionary, whose keys are the available
        """
        methods =  [method for method in dir(self) if callable(getattr(self, method))]
        methods.remove('clustering_algorithms_available')
        methodDict = {}
        for method in methods:
            if not re.match('__', method):
                methodDict[method] = ''
        return methodDict

    def kmeans(self):
        """
            skc.KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)
            Default Parameters:
                params['init'] = 'k-means++'
                params['n_init'] = 10
                params['max_iter'] = 300
                params['tol'] = 0.0001
                params['precompute_distances'] = 'auto'
                params['verbose'] = 0
                params['random_state'] = None
                params['copy_x'] = True
                params['n_jobs'] = 1

        """
        params={}
        params['init'] = 'k-means++'
        params['n_init'] = 10
        params['max_iter'] = 300
        params['tol'] = 0.0001
        params['precompute_distances'] = 'auto'
        params['verbose'] = 0
        params['random_state'] = None
        params['copy_x'] = True
        params['n_jobs'] = 1
        #for anything in self.var_params that may replace defaults, update the param list
        params = returnParams(self.var_params, params, 'kmeans')
        solution=skc.KMeans(n_clusters=self.K, init=params['init'], 
            n_init=params['n_init'], max_iter=params['max_iter'], tol=params['tol'],
            precompute_distances=params['precompute_distances'], verbose=params['verbose'],
            random_state=params['random_state'], copy_x=params['copy_x'], n_jobs=params['n_jobs'])
        solution.fit(self.data)
        self.out = solution.labels_
        self.var_params = params #update dictionary of parameters to match that used.


    def spectral(self):
        """ 
        Calls skc.SpectralClustering()
                solution = skc.SpectralClustering(n_clusters=self.K, n_neighbors=params['n_neighbors'],
                        eigen_solver=params['eigen_solver'], random_state=['random_state'], n_init=params['n_init'],
                        affinity=params['affinity'], 
                        eigen_tol=params['eigen_tol'], assign_labels=params['assign_labels'])
                DEFAULTS:
                        params['eigen_solver']=None
                        params['random_state'] = None
                        params['n_init'] = 10
                        params['affinity'] = 'rbf'
                        params['n_neighbors'] = 10
                        params['eigen_tol'] = '0.0'
                        params['assign_labels'] = 'kmeans'

        """
        params = {}

        params['eigen_solver']=None
        params['random_state'] = None
        params['n_init'] = 10
        params['gamma'] = 1.
        params['affinity'] = 'rbf'
        params['n_neighbors'] = 10
        params['eigen_tol'] = '0.0'
        params['assign_labels'] = 'kmeans'
        params['degree'] = 3
        params['coef0'] = 1
        params['kernel_params']=None

        #for anything in self.var_params that may replace defaults, update the param list
        params = returnParams(self.var_params, params, 'spectral')
 
        solution = skc.SpectralClustering(n_clusters=self.K, n_neighbors=params['n_neighbors'], gamma=params['gamma'],
                        eigen_solver=params['eigen_solver'], random_state=params['random_state'], n_init=params['n_init'],
                        affinity=params['affinity'], coef0=params['coef0'], kernel_params=params['kernel_params'],
                        eigen_tol=params['eigen_tol'], assign_labels=params['assign_labels'])
        solution.fit(self.data)
        self.out = solution.labels_
        self.var_params = params #update dictionary of parameters to match that used.


    def agglomerative(self):
        """
        This calls:
        sklearn.cluster.AgglomerativeClustering(n_clusters=2, affinity='euclidean', connectivity=None, 
        n_components=None, compute_full_tree='auto', linkage='ward', pooling_func=<function mean>)
                DEFAULTS:
                params['affinity'] = 'euclidean'
                params['connectivity']= None
                params['n_components'] = None
                params['compute_full_tree'] = auto
                params['linkage'] = 'ward'
                params['pooling_func'] = np.mean
        """
        params = {}
        params['distance'] = 'euclidean'
        params['affinity'] = params['distance']
        #params['memory'] = 'Memory(cachedir=None)'
        params['connectivity']= None
        params['n_components'] = None
        params['compute_full_tree'] = 'auto'
        params['linkage'] = 'ward'
        params['pooling_func'] = np.mean

        params = returnParams(self.var_params, params, 'agglomerative')
        # reset the distance parameter for 
        params['affinity'] = params['distance'] #if self.var_params had a distance, have to reset here.
        solution = skc.AgglomerativeClustering(n_clusters=self.K, affinity=params['affinity'],
            connectivity=params['connectivity'], n_components= params['n_components'],
            compute_full_tree=params['compute_full_tree'], linkage=params['linkage'] , pooling_func=params['pooling_func'])
        solution.fit(self.data)
        self.out = solution.labels_
        self.var_params = params #update dictionary of parameters to match that used.

    def DBSCAN(self):
        """
        sklearn.cluster.DBSCAN(eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', leaf_size=30, p=None, random_state=None)
            DEFAULTS:
                params['eps']=0.5
                params['min_samples']=5
                params['metric']='euclidean'
                params['algorithm']='auto'
                params['leaf_size']=30
                params['p']=None, 
                params['random_state']=None

        """
        params = {}
        params['distance'] = 'euclidean'
        params['eps']=0.5
        params['min_samples']=5
        params['metric']=params['distance']
        params['algorithm']='auto'
        params['leaf_size']=30
        params['p']=None, 
        params['random_state']=None


        params = returnParams(self.var_params, params, 'DBSCAN')
        params['metric']=params['distance'] #if self.var_params had a distance, have to reset here.

        solution = skc.DBSCAN(eps=params['eps'], min_samples=params['min_samples'], metric=params['metric'], 
            algorithm=params['algorithm'], leaf_size=params['leaf_size'], 
            p=params['p'], random_state=params['random_state']) 
        solution.fit(self.data)
        self.out = solution.labels_
        self.var_params = params #update dictionary of parameters to match that used.

    def AffinityPropagation(self):
        """
        sklearn.cluster.AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='euclidean', verbose=False)
        """
        params = {}
        params['distance'] = 'euclidean'
        params['affinity'] = params['distance']
        params['damping'] = 0.5
        params['max_iter'] = 200
        params['convergence_iter'] = 15
        params['copy'] = True
        params['preference'] = None

        params['verbose'] = False
        params = returnParams(self.var_params, params, 'AffinityPropagation')
        params['affinity'] = params['distance'] #if self.var_params had a distance, have to reset here.
        solution = skc.AffinityPropagation(damping=params['damping'], max_iter=params['max_iter'], convergence_iter=params['convergence_iter'], 
            copy=params['copy'], preference=params['preference'], affinity=params['affinity'], verbose=params['verbose'])
        solution.fit(self.data)
        self.out = solution.labels_
        self.var_params = params #update dictionary of parameters to match that used.

def returnParams(paramsSent, paramsExpected, algorithm):
    """
    A utility for variable parameter setting in clustering algorithms
    Takes two dictionaries of parameter key, value pairs and replaces that in paramsExpected 
    with anything in paramsSent. Will warn users if a key in sent does not appear in expected.
    """
    overlap = set(paramsSent.keys()) & set(paramsExpected.keys())
    params = paramsExpected
    paramsToCheck = paramsSent
    for key in overlap:
        params[key] = paramsSent[key] #if it was sent, overwrite default.
        del paramsToCheck[key]

    #warn if there are keys in paramsToCheck (means they were sent, but not expected) (covers distance metric as well)
    for key in paramsToCheck:
        warnings.warn("Parameter %s was not expected in algorithm %s and will be ignored"%(key, algorithm), UserWarning)

    # Warn if the paramsSent contain a distance, but paramsExpected does not
    #if 'distance' in paramsSent:
    #    if 'distance' not in paramsExpected:
    #        warnings.warn("Algorithm does not take a distance metric and distance metric %s will be ignored"%(paramsSent['distance']), UserWarning)

    return params

