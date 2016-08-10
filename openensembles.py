"""
OpenEnsembles is a resource for performing and analyzing ensemble clustering
"""

import numpy as np 
import pandas as pd 
import sklearn.cluster as skc
import matplotlib.pyplot as plt
from sklearn import datasets
import scipy.cluster.hierarchy as sch


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
            print "ERROR: Size of x-values (%d) does not match that of of the dataframe dimensions (%d), replacing with an vector of integers of correct size"%(len(x), self.D['parent'].shape[1])
            self.x['parent'] = list(range(self.D['parent'].shape[1]))

    def transform(self, source_name, txfm_fcn, txfm_name, var_params):
        """
        This runs transform (txfm_fcn) on the data matrix defined by
        source_name with parameters that are variable for each transform. 
        For example, oe.data.transform('parent', 'zscore','zscore_parent', []) will run the
        zscore in a vector-wise manner across the matrix and the new data
        dictionary access to the transformed data is oe.data['zscore_parent']
        Returns an ERROR -1 if not performed. Successful completion results in
        the addition of a new entry in the data dictionary with a key according
        to txfm_name.
        
        """
        txfm_fcn_dict = {'zscore':0}  
        #CHECK that the source exists
        if source_name not in self.D:
            print "ERROR: the source you requested for transformation does not exist by that name %s"%(source_name)
            return -1

        if txfm_name == 'zscore':
            return 1
                 

        else:
            print "ERROR: the transform function you requested does not exist, currently the following are supported %s"%(txfm_fcn_dict.keys)
            return -2
