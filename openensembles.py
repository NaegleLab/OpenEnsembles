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
#from transforms import transforms
import transforms as tx

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
        TXFM_FCN_DICT = {'zscore':'zscore in a row-wise fashion', 
                'minmax':'scale entire dataset between 0 and 1 or defined minValue and maxValue'}  
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
        
        """
        #CHECK that the source exists
        if source_name not in self.D:
            raise ValueError("ERROR: the source you requested for transformation does not exist by that name %s"%(source_name))
        TXFM_FCN_DICT = self.transforms_available()
        Keep_NaN_txfm = 1 #default value is to keep a transform, even if NaN values are created
        paramDict = {}

        if not kwargs:
            var_params = []
        else:
            var_params = kwargs
            if 'Keep_NaN' in kwargs:
                Keep_NaN_txfm = kwargs['Keep_NaN']

        ######BEGIN TXFM BLOCK  ######
        if txfm_fcn not in TXFM_FCN_DICT:
            raise ValueError( "The transform function you requested does not exist, currently the following are supported %s"%(TXFM_FCN_DICT.keys()))

        txfm = tx.transforms(self.x[source_name], self.D[source_name], kwargs)
        if txfm_fcn == 'zscore':
                txfm.zscore()
        elif txfm_fcn == 'minmax':
                txfm.minmax()

                 

        #### FINAL staging, X, D and var_params have been set in transform block, now add each
        #check and print a warning if NaN values were created in the transformation
        
        boolCheck = np.isnan(txfm.data_out)
        numNaNs = sum(sum(boolCheck))
        if numNaNs:
            print "WARNING: transformation %s resulted in %d NaN values"%(txfm_fcn, numNaNs) 
            if not Keep_NaN_txfm:
                print "Transformation %s resulted in %d NaN values, and you requested not to keep a transformation with NaNs"%(txfm_fcn, numNaNs) 
                return
        self.x[txfm_name] = txfm.x_out 
        self.params[txfm_name] = txfm.var_params
        self.D[txfm_name] = txfm.data_out


