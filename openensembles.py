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
import validation as val
import warnings
from random import randint
import openensembles as oe
from matplotlib.pyplot import cm
from mpl_toolkits.mplot3d import Axes3D

class data:
    """
    The data class has an initialization (taking a dataframe df and x values)

    :raises MyError: Error of the size of x and dimensionality of df do not match

    """
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
        """
        Returns a list of all transformations available
        """
        txfm = tx.transforms(self.x, self.D, {})
        TXFM_FCN_DICT = txfm.transforms_available()
        return TXFM_FCN_DICT

    def plot_data(self, source_name, fig_num=1, **kwargs):
        """ Plot the data matrix that belongs to source_name
            fig_num can be set to a different figure number
            Variable arguments:
                class_labels: this is a vector that assigns points to classes, and will be used to color the plots differently
                fig_num: a different figure number, default ==1
                title 

        """
        m = self.D[source_name].shape[0] 
        n = self.D[source_name].shape[1]
        if 'title' in kwargs:
            title = kwargs['title']
        else:
            title = ''
        if 'class_labels' not in kwargs:
            class_labels = np.ones(m)
        else:
            class_labels = kwargs['class_labels']
        if 'clusters_to_plot' not in kwargs:
            clusters = np.unique(class_labels)
        else:
            #check, this argument should be a list with a direct set match
            clustersPossible = np.unique(class_labels)
            clusters = kwargs['clusters_to_plot']
            if not set(clusters) < set(clustersPossible):
                raise ValueError("ERROR: the clusters to plot are not a direct match to possible clusters in plot_data")



        #clusters = np.unique(class_labels)

        color=iter(cm.rainbow(np.linspace(0,1,len(clusters))))
        
        fig = plt.figure(fig_num, figsize=(6, 6))
        #plt.hold(True)

        #SETUP axes, as either 2D or 3D
        if n==3:
            ax = fig.gca(projection='3d')
        else:
            ax = fig.add_subplot(111)
        #ax.hold(True)

        if n <= 3: #scatter plots for less than 4-dimensions
            for clusterNum in clusters:
                indexes = np.where(class_labels==clusterNum)
                if n==2:
                    plt.scatter(self.D[source_name][indexes,0], self.D[source_name][indexes,1], c=next(color))
                elif n==3:
                    ax.scatter(self.D[source_name][indexes,0], self.D[source_name][indexes,1], self.D[source_name][indexes,2], c=next(color), s=10)


            plt.xlabel(self.x[source_name][0])
            plt.ylabel(self.x[source_name][1])
            if n==3:
                ax.set_zlabel(self.x[source_name][2])

        else:
            plt.hold(True)
            ax.hold(True)
            for clusterNum in clusters:
                indexes = np.where(class_labels==clusterNum)
                plt.plot(self.x[source_name], self.D[source_name][indexes].transpose(), c=next(color))


        plt.title(title)
        #plt.show()
        return fig





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
            warnings.warn("WARNING: transformation %s resulted in %d NaN values"%(txfm_fcn, numNaNs), UserWarning) 
            if not Keep_NaN_txfm:
                #print("Transformation %s resulted in %d NaN values, and you requested not to keep a transformation with NaNs"%(txfm_fcn, numNaNs)) 
                return
        infCheck = np.isinf(txfm.data_out)
        numInf = sum(sum(infCheck))
        if numInf > 0:
            warnings.warn("WARNING: transformation %s resulted in %d Inf values"%(txfm_fcn, numInf), UserWarning) 
            if not Keep_Inf_txfm:
                #print("Transformation %s resulted in %d Inf values, and you requested not to keep a transformation with infinite values"%(txfm_fcn, numInf)) 
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

    def cluster(self, source_name, algorithm, output_name, K=None, Require_Unique=0, **kwargs):

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
        if K: #check if K was overwritten
            if len(uniqueClusters) != K:
                warnings.warn("Number of unique clusters %d returned does not match number requested %d for solution: %s"%(len(uniqueClusters), K, output_name), UserWarning)


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
        params = {}
        params['iterations'] = iterations
        params['K'] = K

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
        name = 'mixture_model'
        c.labels[name] = mixtureObj.labels
        c.data_source[name] = 'parent'
        c.clusterNumbers[name] = np.unique(c.labels[name])
        c.params[name] = params
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
        params={}
        params['linkage'] = linkage
        params['threshold'] = threshold
        coMatObj = self.co_occurrence_matrix('parent')
        coL = finish.co_occurrence_linkage(coMatObj, threshold, linkage=linkage)
        coL.finish()
        c = oe.cluster(self.dataObj)
        name = 'co_occ_linkage'
        c.labels[name] = coL.labels
        c.params[name] = params
        c.data_source[name] = 'parent'
        c.clusterNumbers[name] = np.unique(c.labels[name])
        return c

    def finish_graph_closure(self, threshold, clique_size = 3):
        """ 
        The finishing technique that treats the co-occurrence matrix as a graph, that is binarized by the threshold (>=threshold 
        becomes an unweighted, undirected edge in an adjacency matrix). This graph object is then subjected to clique formation
        according to clique_size (such as triangles if clique_size=3). The cliques are then combined in the graph to create unique
        cluster formations. 
        """
        params = {}
        params['threshold'] = threshold
        params['clique_size'] = clique_size
        coMatObj = self.co_occurrence_matrix('parent')

        c_G = finish.graph_closure(coMatObj.co_matrix, threshold, clique_size=clique_size)
        c_G.finish()
        c = oe.cluster(self.dataObj)
        name = 'graph_closure'
        c.labels[name] = c_G.labels
        c.params[name] = params
        c.data_source[name] = 'parent'
        c.clusterNumbers[name] = np.unique(c.labels[name])
        return c

    def finish_majority_vote(self, threshold=0.5):
        """

        Based on Ana Fred's 2001 paper: Fred, Ana. Finding Consistent Clusters in Data Partitions.
        In Multiple Classifier Systems, edited by Josef Kittler and Fabio Roli, LNCS 2096., 309–18. Springer, 2001.
        This algorithm assingns clusters to the same class if they co-cluster at least 50% of the time. It 
        greedily joins clusters with the evidence that at least one pair of items from two different clusters co-cluster 
        a majority of the time. Outliers will get their own cluster
        """
        params = {}
        coMatObj = self.co_occurrence_matrix('parent')
        c_MV = finish.majority_vote(coMatObj.co_matrix, threshold)
        c_MV.finish()

        c = oe.cluster(self.dataObj)
        name = 'majority_vote'
        c.labels[name] = c_MV.labels
        c.params[name] = params
        c.data_source[name] = 'parent'
        c.clusterNumbers[name] = np.unique(c.labels[name])
        return c


    def get_cluster_members(self, solution_name, clusterNum):
        """ Return the dataframe row indexes of a cluster number in solution named by solution_name """
        indexes = np.where(self.labels[solution_name]==clusterNum)
        return indexes



class validation:
    """
    validation is a class to calculate any number of validation metrics on clustering solutions in data. 
    An individual validation metric must be called on a particular instantiation of the data matrix (like 'parent' or 'zscore')
    and a specific solution in cObj. 
    """
    def __init__(self, dataObj, cObj):
        """ instantiate the object to create a dictionary of validation measurements """
        self.dataObj = dataObj 
        self.cObj = cObj
        self.validation = {} #key here is the name like HC_parent for hierarchically clustered parent
        self.source_name = {}
        self.cluster_name = {}
        self.description = {} #here is a quick description of the validation metric

    def validation_metrics_available(self):
        """ Return all available validation metrics """ 
        validation = val.validation(self.dataObj.D['parent'], [])
        FCN_DICT = validation.validation_metrics_available()
        return FCN_DICT


    def calculate(self, validation_name, cluster_name, source_name='parent'):
        """
        Calls the function titled by validation_name on the data matrix set by source_name (default 'parent') and clustering solution by cluster_name
        Appends to validation with key value equal to the validation_name+source_name+cluster_name
        """

        output_name = "%s_%s_%s"%(validation_name, source_name, cluster_name)

        #check that the validation has not already been calcluated
        if output_name in self.validation:
            warnings.warn('Validation of type requested already exists and will not be added to validation dictionary', UserWarning)
            return

        #CHECK that the source exists
        if source_name not in self.dataObj.D:
            raise ValueError("ERROR: the source you requested for validation does not exist by that name %s"%(source_name))
        if cluster_name not in self.cObj.labels:
            raise ValueError("ERROR: the clustering solution you requested for validation does not exist by the name %s"%(source_name))
        
        FCN_DICT = self.validation_metrics_available()
        
        if validation_name not in FCN_DICT:
            raise ValueError( "The validation metric you requested does not exist, currently the following are supported %s"%(list(FCN_DICT.keys())))
 
        v = val.validation(self.dataObj.D[source_name], self.cObj.labels[cluster_name])
        func = getattr(v,validation_name)
        func()
 
        
        self.validation[output_name] = v.validation
        self.description[output_name] = v.description
        self.source_name[output_name] = source_name
        self.cluster_name[output_name] = cluster_name




