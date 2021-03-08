"""
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


import matplotlib.pyplot as plt
import numpy as np
import pylab
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial import distance as ssd

class coMat:
    """
    A class that allows you to create and operate on a co-occurrence matrix

    Parameters
    ----------
    cObj: openensembles.cluster object
        The clustering object and all contained solutions of interest
    data_source_name: string
        The name of the data source of interest 

    Atrributes
    ----------
    parg: array of ints
        An array reshaped from all contained clustering solutions
    N: int
        Number of objects
    nEnsembles: int
        Number of clustering solutions
    co_matrix: pandas dataframe
        The co-occurrence matrix (square). An entry indicates the fraction of times any pair of objects co-clusters across the ensemble
    avg_dist: float
        The mean of all co-occurrences (not including self distances)


    See Also
    --------
    openensembles.cluster.co_occurrence_matrix()

    """

    def __init__(self, cObj, data_source_name):
        self.cObj = cObj
        self.data_source_name = data_source_name
        parg = []
        for solution in cObj.labels:
            parg.append(cObj.labels[solution])
        self.parg = parg
        self.data = cObj.dataObj.D[data_source_name]
        self.N = self.data.shape[0]
        self.nEnsembles = len(self.parg)
        co_matrix = self.gather_partitions()
        self.co_matrix = co_matrix
        self.avg_dist = np.mean(ssd.squareform(1-self.co_matrix))

    def gather_partitions(self):
        """
        Gather partitions sums the number of times all pairs of objects fall within the same cluster
        across the ensemble.  

        Returns
        -------
        co_matrix_df: pandas dataframe
            a dataframe of object names in column and header, wrapped around the co-occurrence matrix

        todo:: Check that the solution dimensionality and the data matrix dimensions are the same

        """
        dim = self.N
        co_matrix = np.zeros(shape=(dim,dim))
        for solution in self.parg:
            co_bin = self.gather_single_partition(solution)
            co_matrix += co_bin
        co_matrixF = co_matrix/self.nEnsembles
        header = self.cObj.dataObj.df.index.values
        co_matrix_df = pd.DataFrame(index=header, data=co_matrixF,
                columns=header)
        return co_matrix_df

    def gather_single_partition(self, solution):
        """
        For an individual solution (set of labels), create a binary cooccurrence matrix that has an entry of 1 if 
        both objects are in the same cluster and 0 if not. 

        Parameters
        ----------
        Solution: list of ints
            A single solution vector of clustering labels
        
        Returns
        -------
        co_matrix: matrix
            a square matrix the size of the length of solution with boolean values (0,1)


        """
        dim = len(solution)
        co_matrix = np.zeros(shape=(dim,dim))
        clusterid_list = np.unique(solution)
        #print clusterid_list
        for clusterid in clusterid_list:
            itemindex = np.where(solution==clusterid)
            for i,x in enumerate(itemindex[0][0:]):
                co_matrix[x,x] += 1           
                for j,y in enumerate(itemindex[0][i+1:]):
                    co_matrix[x,y]+=1
                    co_matrix[y,x]+=1
        return co_matrix
    
    def pairwise_list(self):
        """
        Reshapes the co-occurrence dataframe matrix into a list, so it can easily be ordered and explored
        
        Returns
        -------
        df: pandas dataframe 
            with a row entry index of object1_object2 and a co-occurrence column labled 'pairwise'

        """
        #return a new dataframe in list with index equal to the pairs being
        #considered 
        coMat = self.co_matrix
        df = pd.DataFrame(columns=['site_1', 'site_2', 'pairwise'])
        headerList = list(coMat)

        for i in range(0, len(headerList)-1):
            for x in range(i+1, len(headerList)):
                s = pd.Series()
                h = headerList[i]
                j = headerList[x]
                s['pairwise'] = coMat.loc[h,j]
                s['site_1'] = h
                s['site_2'] = j
                s.name = '%s; %s'%(h,j)
                df = df.append(s)
        return df

    def link(self, linkage='average'):
        """
        Link a co-occurrence matrix. This is required so that co-occurrence is properly treated as a distance matrix
        during scipy.cluster.hierarchy.linkage

        Parameters
        ----------
        linkage: string
            type of linkage to use, see scipy.cluster.hierarchy.linkage for options

        Returns
        -------
        lnk: scipy.cluster.hierarch.linkage object
        """
        #arr = self.co_matrix
        #set diagonal to zero
        arr = 1 - self.co_matrix
        lnk = sch.linkage(ssd.squareform(arr), method=linkage, metric='euclidean')
        return lnk

    def cut(self, lnk, threshold):
        """
        **A finishing technique**

        Given the calculation of a linkage (e.g. self.lnk(linkage='average')), cut the resulting linkage
        at the given threshold and return the labels from the resulting cut on the hierarchically 
            :param lnk: a linkage object, that can be generated by self.link() 
            :type lnk: scipy.cluster.hierarch.linkage object
            :param threshold: the threshold to cut the groups into. See self.plot() to visually explore the cuts at different thresholds.
            :type threshold: float
        
            :returns: labels - vector of ints assigning each object into a group. This is a type of finishing for deterministic clustering from an ensemble. 
        """
        ind = sch.fcluster(lnk, threshold, 'distance')
        return ind

     
    def plot(self, threshold='avg', linkage='average', add_labels= True, **kwargs):#dist_thresh=self.avg_dist):
        """
        Plot the co_occurrence matrix with a dendrogram and heatmap 
        By Default labels=True, set to false to suppress labels in graph
        By default label_vec equal to the index list of the dataObj dataframe. Otherwise, you can pass in an alternate naming scheme, 
        vector length should be the same as 

        Parameters
        ----------
        threshold: float
            Use threshold to color the dendrogram
            This is useful for identifying visually how to call .cut()
            Default is the average value in the co-occurrence matrix, which is updated to float when 'avg' is passed
        add_labels: bool
            If you wish to shut off printing of labels pass False, else this will print labels according to the co-matrix data frame headers
        linkage: string
            Linkage type to use for dendrogram. Default is average


        Other Parameters
        ----------------

        label_vec: list
            If you want to add labels, but not the same in co-occurrence matrix dataframe, then pass those here

        
        Raises
        ------
            ValueError: 
                if label_vec in **kwargs is different size then number of objects

        Examples
        --------
        >>> coMat = c.co_occurrence_matrix()
        >>> coMat.plot(threshold=1, linkage='average', labels=False)


        """
        if isinstance(threshold, str):
            threshold = self.avg_dist
        
        if add_labels:
            if "label_vec" in kwargs: # use this if you have different labels than in c.dataObj.df.index.values
                label_vec = kwargs['label_vec']
                if len(label_vec) != len(self.co_matrix):
                    raise ValueError("ERROR: the length of label vector does not equal the number of objects in the co_occurrence matrix")
            else:
                label_vec = self.cObj.dataObj.df.index.values.tolist() #using parent just to get column names
        else: 
            label_vec = []

        fig = plot_matrix_sorted(self.co_matrix, label_vec, threshold, self.link(linkage=linkage))
            
        return fig

def plot_matrix_sorted(matrix, label_vec, threshold, lnk1):
    """
    A heatmap plotting function, for both co-occurrence and mutual information

    Parameters
    ----------
    matrix: np.array
        An array of values to plot as a heatmap
    label_vec: list of strings
        A label_vec to label the row-wise objects. Empty if labels not requested
    threshold: float
        Threshold to use for coloring of dendrogram
    lnk1: linkage object
        A pre-calcluated linkage object to use to sort.

    Returns
    -------
    fig: matplotlib.pyplot figure 
        The figure handle 

    """
    fig = pylab.figure(figsize=(10,10))
    panel3 = fig.add_axes([0,0,1,1])
    panel3.axis('off')

    # Add dendrogram 
    if label_vec:
        add_labels = True
    else:
        add_labels = False
    
    if add_labels:
        ax1 = add_subplot_axes(panel3,[0.0,0.3,0.11,.6])
        Z_pp = sch.dendrogram(lnk1, orientation='left', color_threshold=threshold, labels=label_vec)
    else:
        ax1 = add_subplot_axes(panel3,[0.16,0.3,0.11,.6])
        Z_pp = sch.dendrogram(lnk1, orientation='left', color_threshold=threshold)
        ax1.set_yticks([])
    idx_pp = Z_pp['leaves']
    #
    fig.gca().invert_yaxis() # must couple with matshow origin='upper',
    ax1.set_xticks([])
    for side in ['top','right','bottom','left']:
        ax1.spines[side].set_visible(False)

     # plot heatmap
    axmatrix = add_subplot_axes(panel3,[0.28,0.3,0.7,.6])
    hm = matrix
    hm = hm.iloc[idx_pp,idx_pp]
    im = axmatrix.matshow(hm, aspect='auto', origin='upper', cmap='afmhot')
    axmatrix.axis('off')



     # Plot colorbar indicating scale
    axcolor = add_subplot_axes(panel3,[0.28,0.2,0.7,.02]) # [xmin, ymin, dx, and dy]
    h=pylab.colorbar(im, cax=axcolor,orientation='horizontal')
    h.ax.tick_params(labelsize=10)
    h.set_ticks([0.0,.25,.50,.75,1])
    #h.set_ticklabels(['0%','25%','50%','75%','100%'])

    #plt.show()
    return fig


def add_subplot_axes(ax,rect,facecolor='w'):
    """
    A non-class function to handle subaxes

    """
    fig = plt.gcf()
    box = ax.get_position()
    width = box.width
    height = box.height
    inax_position  = ax.transAxes.transform(rect[0:2])
    transFigure = fig.transFigure.inverted()
    infig_position = transFigure.transform(inax_position)    
    x = infig_position[0]
    y = infig_position[1]
    width *= rect[2]
    height *= rect[3]
    subax = fig.add_axes([x,y,width,height],facecolor=facecolor)

    return subax
