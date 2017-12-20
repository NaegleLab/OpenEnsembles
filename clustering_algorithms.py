"""
OpenEnsembles is a resource for performing and analyzing ensemble clustering
This file contains calls to clustering algorithms. Please refer to this 
documentation for specifics about the variable parameters and their defaults, but interact with clustering 
through the openensembles.clustering() class. 
"""

import numpy as np 
import pandas as pd 
import sklearn.cluster as skc
import sklearn as sk
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy.cluster.hierarchy as sch
from sklearn import preprocessing
import scipy.stats as stats
from types import FunctionType
import re
import warnings

class clustering_algorithms:
    """
    The calls to all clustering algorithms reside here. Each call takes on the form of explicitly encoding the default 
    sklearn parameters, overwriting any passed in as kwargs.

    The act of clustering creates a list of assignments of objects in data assigned to clustering classes
    var_params contains all the final parameters used in the act of clustering

    Parameters
    ----------
    data: matrix
        Data matrix, assumes feature dimensions are found column wise (objects in rows)
    kwargs: dict
        Variable parameters passed in as a dict. These are specific to each algorithm
    K: int
        Number of clusters to create, required for most, but not all algorithms. Default K=2

    Attributes
    ----------
    var_params: dict
        The dict of all parameters, including defaults assumed and those overwritten by kwargs

    See Also
    --------
    openensembles.cluster()
    """
    def __init__(self, data, kwargs, K=2):
        self.data = data
        self.out = []
        self.var_params = kwargs
        self.K = K
        #args should have K, even if a default value (should it? not all algorithms need this, use default)
        #if 'K' not in self.args:
        #    raise ValueError('clustering_algorithms should have an instantiated K as part of kwargs key, pair')

    def clustering_algorithms_available(self):
        """
        Report available algorithms
        
        Returns
        -------
        methods: dict
            Dictionary with keys equal to available function calls

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
        kmeans clustering see `skc.KMeans <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html>`_

        **Defaults and var_params:** skc.KMeans(init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='auto', verbose=0, random_state=None, copy_x=True, n_jobs=1)

    
        Other Parameters
        ----------------
        var_params: dict
            Pass variable params through constructor as dictionary pairs. Current default parameters are listed above

        Returns
        -------
        labels: list of ints
            Solution of clustering labels for each object
            
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
        if not self.K:
            raise ValueError('kmeans clustering requires an argument K=<intiger value>')
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
        Spectral clustering, see `skc.SpectralClustering <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.SpectralClustering.html>`_

        **Defaults and var_params:** skc.SpectralClustering(n_clusters=8, eigen_solver=None, random_state=None, n_init=10, gamma=1.0, affinity=’rbf’, n_neighbors=10, eigen_tol=0.0, assign_labels=’kmeans’, degree=3, coef0=1, kernel_params=None, n_jobs=1)
        Other Parameters
        ----------------
        var_params: dict
            Pass variable params through constructor as dictionary pairs. Current default parameters are listed above

        Returns
        -------
        labels: list of ints
            Solution of clustering labels for each object
       
        """
        params = {}

        params['eigen_solver']=None
        params['random_state'] = None
        params['n_init'] = 10
        params['gamma'] = 1.0
        params['affinity'] = 'rbf'
        params['n_neighbors'] = 10
        params['eigen_tol'] = '0.0'
        params['assign_labels'] = 'kmeans'
        params['degree'] = 3
        params['coef0'] = 1
        params['kernel_params']=None
        params['n_jobs']=1

        if not self.K:
            raise ValueError('kmeans clustering requires an argument K=<intiger value>')

        #for anything in self.var_params that may replace defaults, update the param list
        params = returnParams(self.var_params, params, 'spectral')
 
        solution = skc.SpectralClustering(n_clusters=self.K, n_neighbors=params['n_neighbors'], gamma=params['gamma'],
                        eigen_solver=params['eigen_solver'], random_state=params['random_state'], n_init=params['n_init'],
                        affinity=params['affinity'], coef0=params['coef0'], kernel_params=params['kernel_params'],
                        eigen_tol=params['eigen_tol'], assign_labels=params['assign_labels'], n_jobs=params['n_jobs'])
        solution.fit(self.data)
        self.out = solution.labels_
        self.var_params = params #update dictionary of parameters to match that used.

######### BELOW HERE ARE ALGORITHMS that accept DISTANCE METRICS

    def agglomerative(self):
        """
        Uses sklearns `AgglomerativeClustering algorithm <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html>`_ 

        **Defaults and var_params:** sklearn.cluster.AgglomerativeClustering(n_clusters=2, affinity='euclidean', connectivity=None, compute_full_tree='auto', linkage='ward', pooling_func=<function mean>)

        Other Parameters
        ----------------
        var_params: dict
            Pass variable params through constructor as dictionary pairs. Current default parameters are listed above

        Returns
        -------
        labels: list of ints
            Solution of clustering labels for each object
  
        """
        params = {}
        params['distance'] = 'euclidean'
        params['affinity'] = params['distance']
        #params['memory'] = 'Memory(cachedir=None)'
        params['connectivity']= None
        #params['n_components'] = None #gone in newest version
        params['compute_full_tree'] = 'auto'
        params['linkage'] = 'ward'
        params['pooling_func'] = np.mean

        params = returnParams(self.var_params, params, 'agglomerative')
        params['affinity'] = params['distance']

        if not self.K:
            raise ValueError('kmeans clustering requires an argument K=<intiger value>')

        solution = skc.AgglomerativeClustering(n_clusters=self.K, affinity=params['affinity'],
            connectivity=params['connectivity'],
            compute_full_tree=params['compute_full_tree'], linkage=params['linkage'] , pooling_func=params['pooling_func'])
        solution.fit(self.data)
        self.out = solution.labels_
        self.var_params = params #update dictionary of parameters to match that used.


    def DBSCAN(self):
        """
        Uses `sklearn's DBSCAN <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_

        **Defaults and var_params:** sklearn.cluster.DBSCAN(eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', leaf_size=30, p=None, random_state=None)
        
        Other Parameters
        ----------------
        var_params: dict
            Pass variable params through constructor as dictionary pairs. Current default parameters are listed above

        Returns
        -------
        labels: list of ints
            Solution of clustering labels for each object

        
        """
        params = {}
        params['distance'] = 'euclidean'
        params['eps']=0.5
        params['min_samples']=5
        params['metric']='precomputed'
        params['algorithm']='auto'
        params['leaf_size']=30
        params['p']=None, 
        params['n_jobs'] = 1
        params['random_state']=None

        params = returnParams(self.var_params, params, 'DBSCAN')

        #params['distance'] says what to precompute on
        d = returnDistanceMatrix(self.data, params['distance'])
        params['affinity'] = 'precomputed'
        
        #solution = skc.DBSCAN(eps=params['eps'], min_samples=params['min_samples'], metric=params['metric'], 
        #    algorithm=params['algorithm'], leaf_size=params['leaf_size'], 
        #    p=params['p'], random_state=params['random_state']) 
        solution = skc.DBSCAN(eps=params['eps'], min_samples=params['min_samples'], metric=params['metric'], 
            algorithm=params['algorithm'], leaf_size=params['leaf_size'], 
            p=params['p'], n_jobs=params['n_jobs']) #random_state=params['random_state']) 
        solution.fit(d)
        self.out = solution.labels_
        self.var_params = params #update dictionary of parameters to match that used.

    def AffinityPropagation(self):
        """
        Uses `sklearn's Affinity Propagation <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AffinityPropagation.html>`_

        **Defaults and var_params:** sklearn.cluster.AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, affinity='euclidean', verbose=False)

        Other Parameters
        ----------------
        var_params: dict
            Pass variable params through constructor as dictionary pairs. Current default parameters are listed above

        Returns
        -------
        labels: list of ints
            Solution of clustering labels for each object
        """
        params = {}
        params['distance'] = 'euclidean'
        params['affinity'] = 'precomputed'
        params['damping'] = 0.5
        params['max_iter'] = 200
        params['convergence_iter'] = 15
        params['copy'] = True
        params['preference'] = None

        params['verbose'] = False
        params = returnParams(self.var_params, params, 'AffinityPropagation')

        #params['distance'] says what to precompute on
        params['affinity'] = 'precomputed'
        d = returnDistanceMatrix(self.data, params['distance'])
        
        solution = skc.AffinityPropagation(damping=params['damping'], max_iter=params['max_iter'], convergence_iter=params['convergence_iter'], 
            copy=params['copy'], preference=params['preference'], affinity=params['affinity'], verbose=params['verbose'])
        solution.fit(d) #operates on distance matrix

        self.out = solution.labels_
        self.var_params = params #update dictionary of parameters to match that used.

def returnParams(paramsSent, paramsExpected, algorithm):
    """
    A utility for variable parameter setting in clustering algorithms
    Takes two dictionaries of parameter key, value pairs and replaces that in paramsExpected 
    with anything in paramsSent. 

    Returns
    -------
    params: dict
        Dict of parameters that represent the final parameters, overwritten in paramsExpected by paramsSent

    Warnings
    --------
        Will warn users if a key in sent does not appear in expected.
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

def returnDistanceMatrix(data, distance):
    """
    A utility to calculate a distance matrix, according to type in <distance> on the data array. 

    Parameters
    ----------
    data: matrix
        Data matrix to calculate distances from
    distance: string
        Distance metric. See `sklearn's pairwise distances <http://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.pairwise_distances.html>`_

    Returns
    -------
    d: matrix 
        the distance matrix computed by distance

    Raises
    ------
    ValueError:
        if the distance metric is not available.
    """
    distDict = sk.metrics.pairwise.distance_metrics()
    if distance not in distDict:
        raise ValueError("ERROR: the distance you requested is not available by that name %s. Please see sklearn.metrics.pairwise.distance_metrics()"%(distance))
    
    d = distDict[distance](data)

    return d

