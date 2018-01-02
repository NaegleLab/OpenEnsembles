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
from scipy.spatial import distance as ssd


class MI:
	"""
	A class that allows you to calculate a mutual information matrix comparing all clustering solutions


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


