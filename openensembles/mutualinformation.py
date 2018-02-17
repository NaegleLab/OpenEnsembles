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

import sklearn.metrics as skm
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import numpy as np
import pylab
import pandas as pd
import scipy.cluster.hierarchy as sch
import openensembles.cooccurrence as co

from scipy.spatial import distance as ssd


class MI:
	"""
	A class that allows you to calculate a mutual information matrix comparing all clustering solutions
	
	Parameters
	----------
	cObj: an openensembles.cluster object
	    The clustering object and all contained solutions of interest
	MI_type: string {'standard', 'normalized', 'adjusted'}
		The type of mutual information to use

	Attributes
	----------
	matrix: pandas dataframe
		A pandas dataframe with rows and index equal to the solution names in cObj.labels and entries with mutual information
	MI_type: string
		Stores the type of MI object was instantiated with

	See Also
	--------
	openensembles.cluster.MI()



	"""

	def __init__(self, cObj, MI_type):
		self.cObj = cObj
		self.MI_type = MI_type

		#check that MI_type is recognized
		if MI_type != 'standard' and MI_type != 'adjusted' and MI_type != 'normalized':			
			raise ValueError("Did not recognize MI_type %s as one of standard/adjusted/normalized"%(MI_type))

		
		#get all names of solutions in cObj, these we will walk through
		names = list(cObj.labels.keys())
		d = pd.DataFrame(columns=names, index=names)

		for name_1 in names:
			for name_2 in names:
				if np.isnan(d.loc[name_1, name_2]):
					MI = calculate_MI(cObj.labels[name_1], cObj.labels[name_2], MI_type)
					d = d.set_value(name_1, name_2, MI)
					d = d.set_value(name_2, name_1, MI)
				#check that the value is nonexistent, then place calculation in the symmetric positions 


		#before leaving, check that all entries have been calculated
		#if np.where(pd.isnull(d)):
		#	raise ValueError("Problem: not all entries were calculated")

		#self.matrix = pd.Dataframe
		self.matrix = d

	def plot(self, threshold=0, linkage='average', add_labels= True, **kwargs):#dist_thresh=self.avg_dist):
		"""
		Plot the mutual information matrix with a dendrogram and heatmap 
		By Default labels=True, set to false to suppress labels in graph
		By default label_vec equal to the index list of the solution. Otherwise, you can pass in an alternate naming scheme, 
		vector length should be the same as matrix dimension length

		Parameters
		----------
		threshold: float
		    Use threshold to color the dendrogram
		    This is useful for identifying visually how to call .cut()
		add_labels: bool
		    If you wish to shut off printing of labels pass False, else this will print labels according to the co-matrix data frame headers
		linkage: string
		    Linkage type to use for dendrogram. Default is average


		Other Parameters
		----------------
		label_vec: list
		    If you want to add labels, but not the same in mutual information matrix dataframe, then pass those here


		Raises
		------
		    ValueError: 
		        if label_vec in **kwargs is different size then number of objects

		Examples
		--------
		Plot the adjusted Mutual Information, with no labels
		>>> MI = c.MI(MI_type='adjusted')
		>>> MI.plot(threshold=1, linkage='average', labels=False)


		"""
		if add_labels:
		    if "label_vec" in kwargs: # use this if you have different labels than in c.dataObj.df.index.values
		        label_vec = kwargs['label_vec']
		        if len(label_vec) != len(self.matrix):
		            raise ValueError("ERROR: the length of label vector does not equal the number of objects in the co_occurrence matrix")
		    else:
		        label_vec = self.matrix.index.values.tolist()
		else: 
		    label_vec = []

		arr = 1 - self.matrix
		lnk = sch.linkage(ssd.squareform(arr), method=linkage, metric='euclidean')


		fig = co.plot_matrix_sorted(self.matrix.astype('float32'), label_vec, threshold, lnk)
		    
		return fig





def calculate_MI(a,b, MI_type):
	"""
	Calucate the mutual information, according to MI_type, for classes given in lists a and b.

	Parameters
	----------
	a: list of ints
		One set of class assignments of objects 
	b: list of ints
		Another set of class assignments for objects in a

	Returns
	-------
	MI: float
		Mutual information between class assignments in a and b

	"""
	if MI_type == 'standard':
		MI = skm.mutual_info_score(a,b)

	elif MI_type == 'adjusted':
		MI = skm.adjusted_mutual_info_score(a,b)

	elif MI_type=='normalized':
		MI = skm.normalized_mutual_info_score(a,b)

	else:
		raise ValueError("Did not recognize MI_type %s as one of standard/adjusted/normalized"%(MI_type))

	return MI


