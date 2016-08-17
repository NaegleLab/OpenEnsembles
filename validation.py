from sklearn import datasets
import math
from scipy.spatial import distance
import matplotlib.pyplot as plt
import numpy as np
import itertools
import re
import warnings
from sklearn import metrics

class validation:
	"""
	validation is a class for calculating validation metrics on a data matrix, data, given the clustering labels in labels. 
	Instantiation sets validation to NaN and a description to ''. Once a metric is performed, these are replaced (unless)
	validation did not yield a valid mathematical number, which can happen in certain cases, such as when a cluster 
	consists of only one member. Such results will warn the user.
	"""
	def __init__(self, data, labels):
		self.dataMatrix = data
		self.classLabel = labels
		self.validation = np.nan
		self.description = ''

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

	def Ball_Hall(self):
		"""
		Ball-Hall Index is the mean of the mean dispersion across all clusters
		"""
		self.description = 'Mean of the mean dispersions across all clusters'
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


	def Banfeld_Raferty(self):
		""" Banfeld-Raferty index is the weighted sum of the logarithms of the traces of the variance-covariance matrix of each cluster"""
		self.description = 'Weighted sum of the logarithms of the traces of the variance-covariance matrix of each cluster'
		sumTotal=0
		numCluster=max(self.classLabel)+1
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
			if sumDis/len(indices) <= 0:
				warnings.warn('Cannot calculate Banfeld_Raferty, due to an undefined value', UserWarning)
			else:
				sumTotal=sumTotal+len(indices)*math.log(sumDis/len(indices))
		#return the fitness
				self.validation = sumTotal
		return self.validation
		
		## The Baker-HUbert Gamma Index BHG

	def silhouette(self):
		"""
		Silhouette: Compactness and connectedness combination that measures a ratio of within cluster distances to closest neighbors
		outside of cluster.
		"""
		self.description = 'Silhouette: A combination of connectedness and compactness that measures within versus to the nearest neighbor outside a cluster. A smaller value, the better the solution'
		
		metric = metrics.silhouette_score(self.dataMatrix, self.classLabel, metric='euclidean')
		self.validation = metric
		return self.validation 

	def Baker_Hubert_Gamma(self):	
		"""
		Baker-Hubert Gamma Index: A measure of compactness, based on similarity between points in a cluster, compared to similarity 
		with points in other clusters
		"""
		self.description = 'Gamma Index: a measure of compactness'
		splus=0
		sminus=0
		pairDis=distance.pdist(self.dataMatrix)
		numPair=len(pairDis)
		temp=np.zeros((len(self.classLabel),2))
		temp[:,0]=self.classLabel
		vecB=distance.pdist(temp)
		#iterate through all the pairwise comparisons
		for i in range(numPair-1):
			for j in range(i+1,numPair):
				if vecB[i]>0 and vecB[j]==0:
					#heter points smaller than homo points
					if pairDis[i]<pairDis[j]:
						splus=splus+1
					#heter points larger than homo points
					if pairDis[i]>vecB[j]:
						sminus=sminus+1
				if vecB[i]==0 and vecB[j]>0:
					#heter points smaller than homo points
					if pairDis[j]<pairDis[i]:
						splus=splus+1
					#heter points larger than homo points
					if pairDis[j]>vecB[i]:
						sminus=sminus+1
		#compute the fitness
		self.validation = (splus-sminus)/(splus+sminus)
		return self.validation