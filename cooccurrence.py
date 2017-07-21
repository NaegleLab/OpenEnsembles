import matplotlib.pyplot as plt
import numpy as np
import pylab
import pandas as pd
import scipy.cluster.hierarchy as sch
from scipy.spatial import distance as ssd

class coMat:
    '''
        A class that allows you to create and operate on a
        co-occurrence matrix
     '''

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
         dim = self.N
         co_matrix = np.zeros(shape=(dim,dim))
         for solution in self.parg:
             co_bin = self.gather_single_partition(solution)
             co_matrix += co_bin
         co_matrixF = co_matrix/self.nEnsembles
         header = self.cObj.dataObj.df.index.get_values()
         co_matrix_df = pd.DataFrame(index=header, data=co_matrixF,
                 columns=header)
         return co_matrix_df

    def gather_single_partition(self, solution):
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
        Returns a linkage object
        """
        #arr = self.co_matrix
        #set diagonal to zero
        arr = 1 - self.co_matrix
        lnk = sch.linkage(ssd.squareform(arr), method=linkage, metric='euclidean')
        return lnk

    def cut(self, lnk, threshold):
        """
        Given the calculation of a linkage (self.lnk(linkage='average')), cut the resulting linkage
        at the given threshold and return the labels that are the resulting clustering. 
        """
        ind = sch.fcluster(lnk, threshold, 'distance')
        return ind

     
    def plot(self, **kwargs):#dist_thresh=self.avg_dist):
        """
        Plot the co_occurrence matrix, using dist_threshold to color the dendrogram. 
        Uses average linkage to sort the dendrogram. Can plot using this threshold and cut using .cut()
        By Default labels=True, set to false to suppress labels in graph
        By default label_vec equal to the index list of the dataObj dataframe. Otherwise, you can pass in an alternate naming scheme, 
        vector length should be the same as 
        Example:
        cOcc.plot(dist_thresh=0.5)
        """
        if "distance_threshold" in kwargs:
            dist_thresh = kwargs['distance_threshold']
        else:
            dist_thresh = self.avg_dist
        if "labels" in kwargs:
            add_labels = kwargs['labels']
        else:
            add_labels = True
        if "label_vec" in kwargs: # use this if you have different labels than in c.dataObj.df.index.values
            label_vec = kwargs['label_vec']
            if len(label_vec) != len(self.co_matrix):
                raise ValueError("ERROR: the length of label vector does not equal the number of objects in the co_occurrence matrix")
        else:
            label_vec = self.cObj.dataObj.df.index.values.tolist() #using parent just to get column names
            

        fig = pylab.figure(figsize=(10,10))
        panel3 = fig.add_axes([0,0,1,1])
        panel3.axis('off')

        # Add dendrogram 
        
        lnk1 = self.link(linkage='average')
        if add_labels:
            ax1 = add_subplot_axes(panel3,[0.0,0.3,0.11,.6])
            Z_pp = sch.dendrogram(lnk1, orientation='left', color_threshold=dist_thresh, labels=label_vec)
        else:
            ax1 = add_subplot_axes(panel3,[0.16,0.3,0.11,.6])
            Z_pp = sch.dendrogram(lnk1, orientation='left', color_threshold=dist_thresh)
            ax1.set_yticks([])
        idx_pp = Z_pp['leaves']
        #
        fig.gca().invert_yaxis() # must couple with matshow origin='upper',
        ax1.set_xticks([])
        for side in ['top','right','bottom','left']:
            ax1.spines[side].set_visible(False)

         # plot heatmap
        axmatrix = add_subplot_axes(panel3,[0.28,0.3,0.7,.6])
        hm = self.co_matrix
        hm = hm.ix[idx_pp,idx_pp]
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
