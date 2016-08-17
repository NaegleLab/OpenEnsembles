from sklearn import datasets
import math
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
import itertools

class validation:
	def __init__(self, data, labels):
		self.dataMatrix = data
		self.classLabel = labels
		self.validation = NaN

	def validation_metrics_available(self):
		"""
    	self.validation_metrics_available() returns a dictionary, whose keys are the available validation metrics
    	"""
		methods =  [method for method in dir(self) if callable(getattr(self, method))]
		methods.remove('validation_metrics_available')
		methodDict = {}
		for method in methods:
			if not re.match('__', method):
				methodDict[method] = ''
		return methodDict

	## Ball-Hall index BHI
	def Ball_Hall_Index(self):
		"""
		Ball-Hall Index is ... 
		"""
		sumTotal=0

		numCluster=len(np.unique(self.classLabel))
		#iterate through all the clusters
		for i in range(numCluster):
			sumDis=0
			indices=[t for t, x in enumerate(self.classLabel) if x == i]
			clusterMember=self.dataMatrix[indices,:]
			#compute the center of the cluster
			clusterCenter=np.mean(clusterMember,0)
			#iterate through all the members
			for member in clusterMember:
				sumDis=sumDis+math.pow(distance.euclidean(member, clusterCenter),2)
			sumTotal=sumTotal+sumDis/len(indices)
		#compute the validation
		self.validation = sumTotal/numCluster
		return self.validation