"""
OpenEnsembles is a resource for performing and analyzing ensemble clustering
This file contains calls to clustering algorithms. Please refer to this 
documentation for specifics about the variable parameters and their defaults, but interact with clustering 
through the openensembles.clustering() class. 

OpenEnsembles is a resource for performing and analyzing ensemble clustering

Copyright (C) 2017 Naegle Lab

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
        The keyword 'distance' is used by OpenEnsembles to generalize distance/similarity calls across algorithms. Use 'distance' and a string for distance metric. 
        Algorithms requiring similarity will use the conversion of Distance (D) to Similarity (S) according to: S = np.exp(-D / D.std()). 
        Use 'affinity' and 'precomputed' to exert greater control over usage. 
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
            Solution of clustering labels for each object (updated in object.out)
            
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

        seed = params['random_state'][1][0]
        solution=skc.KMeans(n_clusters=self.K, init=params['init'], 
            n_init=params['n_init'], max_iter=params['max_iter'], tol=params['tol'],
            precompute_distances=params['precompute_distances'], verbose=params['verbose'],
            random_state=seed, copy_x=params['copy_x'], n_jobs=params['n_jobs'])
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
            Solution of clustering labels for each object (updated in object.out)       
        """
        params = {}

        params['affinity'] = 'rbf'
        params['assign_labels'] = 'kmeans'
        params['coef0'] = 1
        params['degree'] = 3
        params['eigen_solver']=None
        params['eigen_tol'] = '0.0'
        params['gamma'] = 1.0
        params['kernel_params']=None
        params['n_init'] = 10
        params['n_jobs']=1
        params['n_neighbors'] = 10
        params['random_state'] = None

        #NOt used directly by spectral, the true default is affinity with rbf
        params['distance'] = 'euclidean'
        params['M'] = []

        if not self.K:
            raise ValueError('spectral clustering requires an argument K=<intiger value>')

        #for anything in self.var_params that may replace defaults, update the param list
        params = returnParams(self.var_params, params, 'spectral')
 
        seed = params['random_state'][1][0]

        # handle the cases of affinity set, affinity as precomputed with a matrix, distance as a string that needs to be converted and distance as precomputed, which shoudl fail

        if 'affinity' in self.var_params:
            if self.var_params['affinity'] == 'precomputed':
                solution = skc.SpectralClustering(n_clusters=self.K, n_neighbors=params['n_neighbors'], gamma=params['gamma'],
                    eigen_solver=params['eigen_solver'], random_state=seed, n_init=params['n_init'],
                    affinity='precomputed', coef0=params['coef0'], kernel_params=params['kernel_params'],
                    eigen_tol=params['eigen_tol'], assign_labels=params['assign_labels'], n_jobs=params['n_jobs'])
                x = np.shape(self.var_params['M'])
                solution.fit(self.var_params['M'])
                params['M'] = self.var_params['M']
            else: #it's affinity, that's not precomputed, but overrides the default
                solution = skc.SpectralClustering(n_clusters=self.K, n_neighbors=params['n_neighbors'], gamma=params['gamma'],
                            eigen_solver=params['eigen_solver'], random_state=seed, n_init=params['n_init'],
                            affinity=params['affinity'], coef0=params['coef0'], kernel_params=params['kernel_params'],
                            eigen_tol=params['eigen_tol'], assign_labels=params['assign_labels'], n_jobs=params['n_jobs'])
                solution.fit(self.data)

        elif 'distance' in self.var_params:
            if self.var_params['distance'] == 'precomputed':
                raise ValueError("If precomputing a matrix for Spectral clustering, it must be a similarity matrix")

            params['affinity'] = 'precomputed'
            D = returnDistanceMatrix(self.data, self.var_params['distance'])
            S = convertDistanceToSimilarity(D)
            solution = skc.SpectralClustering(n_clusters=self.K, n_neighbors=params['n_neighbors'], gamma=params['gamma'],
                        eigen_solver=params['eigen_solver'], random_state=seed, n_init=params['n_init'],
                        affinity='precomputed', coef0=params['coef0'], kernel_params=params['kernel_params'],
                        eigen_tol=params['eigen_tol'], assign_labels=params['assign_labels'], n_jobs=params['n_jobs'])
            solution.fit(S)
        else: #else it's an affinity that is not precomputed.
            solution = skc.SpectralClustering(n_clusters=self.K, n_neighbors=params['n_neighbors'], gamma=params['gamma'],
                            eigen_solver=params['eigen_solver'], random_state=seed, n_init=params['n_init'],
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
            Solution of clustering labels for each object (updated in object.out)  
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
            raise ValueError('agglomerative clustering requires an argument K=<intiger value>')

        solution = skc.AgglomerativeClustering(n_clusters=self.K, affinity=params['affinity'],
            connectivity=params['connectivity'],
            compute_full_tree=params['compute_full_tree'], linkage=params['linkage'] , pooling_func=params['pooling_func'])
        solution.fit(self.data)
        self.out = solution.labels_
        self.var_params = params #update dictionary of parameters to match that used.


    def DBSCAN(self):
        """
        Uses `sklearn's DBSCAN <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.DBSCAN.html>`_

        **Defaults and var_params:** sklearn.cluster.DBSCAN(eps=0.5, min_samples=5, metric='euclidean', algorithm='auto', leaf_size=30, p=None, n_jobs=1)
        
        Other Parameters
        ----------------
        var_params: dict
            Pass variable params through constructor as dictionary pairs. Current default parameters are listed above

        Returns
        -------
        labels: list of ints
            Solution of clustering labels for each object (updated in object.out)
        
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

        params = returnParams(self.var_params, params, 'DBSCAN')

        if 'distance' in self.var_params:
            if self.var_params['distance'] == 'precomputed':
                d = self.var_params['M']
        else:
            d = returnDistanceMatrix(self.data, params['distance'])        

        solution = skc.DBSCAN(eps=params['eps'], min_samples=params['min_samples'], metric=params['metric'], 
            algorithm=params['algorithm'], leaf_size=params['leaf_size'], 
            p=params['p'], n_jobs=params['n_jobs']) 
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
            Solution of clustering labels for each object (updated in object.out)
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

    def Birch(self):
        """
        Uses `sklearn's Birch <http://scikit-learn.org/stable/modules/generated/sklearn.cluster.Birch.html>`_
        **Defaults and var_params:** sklearn.cluster.Birch(threshold=0.5, branching_factor=50, compute_labels=True, copy=True)

        Other Parameters
        ----------------
        var_params: dict
            Pass variable params through constructor as dictionary pairs. Current default parameters are listed above

        Returns
        -------
        labels: list of ints
            Solution of clustering labels for each object (updated in object.out)
        """
        params = {}
        params['distance'] = 'euclidean' #not mutable
        params['threshold'] = 0.5
        params['branching_factor'] = 50
        params['n_clusters'] = self.K
        params['compute_labels'] = True
        params['copy'] = True
        
        if not self.K:
            raise ValueError('Birch clustering requires an argument K=<intiger value>')

        params = returnParams(self.var_params, params, 'Birch')
        d = returnDistanceMatrix(self.data, params['distance'])

        solution = skc.Birch(threshold=params['threshold'], branching_factor=params['branching_factor'], n_clusters=params['n_clusters'],
            compute_labels=params['compute_labels'], copy=params['copy'])
        solution.fit(d)

        self.out = solution.labels_
        self.var_params = params


def returnParams(paramsSent, paramsExpected, algorithm):
    """
    A utility for variable parameter setting in clustering algorithms
    Takes two dictionaries of parameter key, value pairs and replaces that in paramsExpected 
    with anything in paramsSent. 

    Returns
    -------
    params: dict
        Dict of parameters that represent the final parameters, overwritten in paramsExpected by paramsSent
        This will handle checking to make sure that if precomputed distances have been selected, that a distance 
        or similarity matrix is also passed.

    Warnings
    --------
        Will warn users if a key in sent does not appear in expected.
    """

    if 'distance' in paramsSent:
        if paramsSent['distance'] == 'precomputed':
            if 'M' not in paramsSent:
                raise ValueError("Precomputed distances require a distance matrix passed as 'M' ")

    if 'affinity' in paramsSent:
        if paramsSent['affinity'] == 'precomputed':
            if 'M' not in paramsSent:
                raise ValueError("Precomputed affinity require a similarity matrix passed as 'M' ")

    overlap = set(paramsSent.keys()) & set(paramsExpected.keys())
    params = paramsExpected.copy()
    paramsToCheck = paramsSent.copy()
    for key in overlap:
        params[key] = paramsSent[key] #if it was sent, overwrite default.
        del paramsToCheck[key]

    #warn if there are keys in paramsToCheck (means they were sent, but not expected) (covers distance metric as well)
    if 'random_state' in paramsToCheck:
        del paramsToCheck['random_state']
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
    #distDict = sk.metrics.pairwise.pairwise_distances()
    #if distance not in distDict:
    #    raise ValueError("ERROR: the distance you requested, %s, is not available. Please see sklearn.metrics.pairwise.distance_metrics()"%(distance))
    
    #d = distDict[distance](data)
    d = sk.metrics.pairwise.pairwise_distances(data, metric=distance)

    return d

def convertDistanceToSimilarity(D, beta=1.0):
    """
    A utility to convert a distance matrix to a similarity matrix

    Parameters
    ----------
    D: matrix of floats
        A matrix of distances, such as returned by returnDistanceMatrix(data,distanceType)
    beta: float
        A variable for mapping distance to similarity.

    Returns
    -------
    S: a matrix of floats
        A matrix of similarity values. according to S = np.exp(-beta * D / D.std())
    """
    S = np.exp(-beta*D/D.std())
    return S

