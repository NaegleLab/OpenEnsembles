"""
OpenEnsembles is a resource for performing and analyzing ensemble clustering
"""
import numpy as np 
import pandas as pd 
import sklearn.cluster as skc
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy.cluster.hierarchy as sch
from sklearn import preprocessing
import scipy.stats as stats
import transforms as tx
import clustering_algorithms as ca 
import finishing as finish
import cooccurrence as co
import warnings
from random import randint
import openensembles as oe

class data:
    
    def __init__(self, df, x):
        """
        df is a dataframe and x is the x_axis values (or numbers indicating the
        number entry)
        """
        self.df = df
        self.D = {}
        self.x = {}
        self.params = {}
        self.D['parent'] = np.asarray(df)
        self.x['parent'] = x
        self.params['parent'] = []

        #check that the number of x-values matches the array
        if(len(x) != self.D['parent'].shape[1]):
            raise ValueError("ERROR: Size of x-values (%d) does not match that of of the dataframe dimensions (%d), replacing with an vector of integers of correct size"%(len(x), self.D['parent'].shape[1]))

    def transforms_available(self):
        txfm = tx.transforms(self.x, self.D, {})
        TXFM_FCN_DICT = txfm.transforms_available()
        return TXFM_FCN_DICT

    def transform(self, source_name, txfm_fcn, txfm_name, **kwargs):
        """
        This runs transform (txfm_fcn) on the data matrix defined by
        source_name with parameters that are variable for each transform. 
        For example, oe.data.transform('parent', 'zscore','zscore_parent', []) will run the
        zscore in a vector-wise manner across the matrix and the new data
        dictionary access to the transformed data is oe.data['zscore_parent']
        Returns an ERROR -1 if not performed. Successful completion results in
        the addition of a new entry in the data dictionary with a key according
        to txfm_name.
        Default Behavior:
                Keep_NaN = 1 (this will add transformed data even if NaNs are produced. Set to 0 to prevent addition of data transforms containing NaNs. 
                Keep_Inf = 1 (this will add transformed data even if infinite values are produced. Set to 0 to prevent addition of data transforms containing Inf. 
        
        """
        #CHECK that the source exists
        if source_name not in self.D:
            raise ValueError("ERROR: the source you requested for transformation does not exist by that name %s"%(source_name))
        TXFM_FCN_DICT = self.transforms_available()
        Keep_NaN_txfm = 1 #default value is to keep a transform, even if NaN values are created
        Keep_Inf_txfm = 1 #default value is to keep a transform, even if NaN values are created
        paramDict = {}

        if not kwargs:
            var_params = []
        else:
            var_params = kwargs
            if 'Keep_NaN' in kwargs:
                Keep_NaN_txfm = kwargs['Keep_NaN']
            if 'Keep_Inf' in kwargs:
                Keep_Inf_txfm = kwargs['Keep_Inf']

        ######BEGIN TXFM BLOCK  ######
        if txfm_fcn not in TXFM_FCN_DICT:
            raise ValueError( "The transform function you requested does not exist, currently the following are supported %s"%(list(TXFM_FCN_DICT.keys())))

        txfm = tx.transforms(self.x[source_name], self.D[source_name], kwargs)
        func = getattr(txfm,txfm_fcn)
        func()
 
        #### FINAL staging, X, D and var_params have been set in transform block, now add each
        #check and print a warning if NaN values were created in the transformation
        
        boolCheck = np.isnan(txfm.data_out)
        numNaNs = sum(sum(boolCheck))
        if numNaNs:
            print("WARNING: transformation %s resulted in %d NaN values"%(txfm_fcn, numNaNs)) 
            if not Keep_NaN_txfm:
                print("Transformation %s resulted in %d NaN values, and you requested not to keep a transformation with NaNs"%(txfm_fcn, numNaNs)) 
                return
        infCheck = np.isinf(txfm.data_out)
        numInf = sum(sum(infCheck))
        if numInf > 0:
            print("WARNING: transformation %s resulted in %d Inf values"%(txfm_fcn, numInf)) 
            if not Keep_Inf_txfm:
                print("Transformation %s resulted in %d Inf values, and you requested not to keep a transformation with infinite values"%(txfm_fcn, numInf)) 
                return

        self.x[txfm_name] = txfm.x_out 
        self.params[txfm_name] = txfm.var_params
        self.D[txfm_name] = txfm.data_out

class cluster:
    def __init__(self, dataObj):
        """
        dataObj is an openensembles.data class, which can consist of many data matrices, but at the 
        very least, consists of 'parent'
        """
        self.dataObj = dataObj 
        self.labels= {} #key here is the name like HC_parent for hierarchically clustered parent
        self.data_source = {} # keep track of the key to the data source in object used
        self.params = {}
        self.clusterNumbers = {}


    def algorithms_available(self):
        algorithms = ca.clustering_algorithms(self.dataObj.D['parent'], {})
        ALG_FCN_DICT = algorithms.clustering_algorithms_available()
        return ALG_FCN_DICT

    def cluster(self, source_name, algorithm, output_name, K=2, Require_Unique=0, **kwargs):

        """
        This runs clustering algorithms on the data matrix defined by
        source_name with parameters that are variable for each algorithm. Note that K is 
        required for most algorithms and is given a default of K=2. 
        For example, clusterObj.cluster('parent', 'kmeans','kmeans_parent', K=5) will run the
        perform k-means clustering, with K=5, on the parent data matrix that belongs to the 
        data object used to instantiate clusterObj = oe.cluster(dataObj) 

        This will warn if the number of clusters is differen than what was requested

        DEFAULT Behavior: 
            This will add a number to a requested output_name, if that name exists already, unless Require_Unique=1
        
        """
        #CHECK that the source exists
        if source_name not in self.dataObj.D:
            raise ValueError("ERROR: the source you requested for clustering does not exist by that name %s"%(source_name))
        ALG_FCN_DICT = self.algorithms_available()
        paramDict = {}

        if not kwargs:
            var_params = {} 
        else:
            var_params = kwargs
        
        ##### Check to see if the same name exists for clustering solution name and decide what to do according to Require_Unique
        if output_name in list(self.labels.keys()):
            if Require_Unique:
                raise ValueError('The name of the clustering solution is redundant and you required unique')
            else:
                test_name = "%s_%d"%(output_name, randint(0,10000))
                while test_name in self.labels:
                    test_name = "%s_%d"%(output_name, randint(0,10000))
                output_name = test_name
                warnings.warn('For uniqueness, altered output_name to be %s'%(output_name), UserWarning)

        ######BEGIN CLUSTERING BLOCK  ######
        if algorithm not in ALG_FCN_DICT:
            raise ValueError( "The algorithm you requested does not exist, currently the following are supported %s"%(list(ALG_FCN_DICT.keys())))


        c = ca.clustering_algorithms(self.dataObj.D[source_name], var_params, K)
        func = getattr(c,algorithm)
        func()
 
        #### FINAL staging, c now contains a finished assignment and c.params has final parameters used.

        # CHECK that K is as requested 
        uniqueClusters = np.unique(c.out)
        if len(uniqueClusters) != K:
            warnings.warn('Number of unique clusters returned does not match number requested', UserWarning)


        self.labels[output_name] = c.out
        self.data_source[output_name] = source_name
        self.params[output_name] = c.var_params
        self.clusterNumbers[output_name] = uniqueClusters

    def co_occurrence_matrix(self, data_source_name):
        """
        coMat = self.co_occurrence_matrix(data_source_name) creates a co_occurrence object that is linked to a particular source of data, such as 'parent'. 
        coMat.co_matrix is an NxN matrix, whose entries indicate the number of times the pair of objects in positon (i,j) cluster across the ensemble
        of clustering labels that exist in self.labels, where labels is a dictionary of solutions 
        """
        coMat = co.coMat(self, data_source_name)
        return coMat


    def mixture_model(self, K=2, iterations=10):
        """
        Operates on entire ensemble of clustering solutions in self, to create a mixture model
        See finishing.mixture_model for more details. This implementation is based on 
        Topchy, Jain, and Punch, "A mixture model for clustering ensembles Proc. SIAM Int. Conf. Data Mining (2004)"
        Returns a new clustering object with c.labels['mixture_model'] set to the final solution. 
        """

        #check to make sure more than one solution exists in ensemble
        if len(self.params) < 2:
            raise ValueError("Mixture Model is a finsihing technique for an ensemble, the cluster object must contain more than one solution")
        N = self.dataObj.D['parent'].shape[0]
        parg = []
        for solution in self.labels:
            parg.append(self.labels[solution])

        mixtureObj = finish.mixture_model(parg, N, nEnsCluster=K, iterations=iterations)
        mixtureObj.emProcess()
        c = oe.cluster(self.dataObj)
        c.labels['mixture_model'] = mixtureObj.labels
        return c

    def finish_co_occ_linkage(self, threshold, linkage='average'):
        """
        The finishing technique that calculates a co-occurrence matrix on all cluster solutions in the ensemble and 
        then hierarchically clusters the co-occurrence, treating it as a similarity matrix. The clusters are defined by 
        the threshold of the distance used to cut. To determine this visually, do the following:
            coMat = c.co_occurrence(linkage=<linkage>)
            coMat.plot(threshold=<threshold>)
        The resulting clusters from a cut made at <threshold> will be colored accordingly.
        """

        coL = finish.co_occurrence_linkage(self, threshold, linkage=linkage)
        coL.finish()
        c = oe.cluster(self.dataObj)
        c.labels['co_occ_linkage'] = coL.labels
        return c


