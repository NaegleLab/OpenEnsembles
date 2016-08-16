import numpy as np
import pandas as pd
import pylab
import re
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
import scipy.spatial.distance as ssd
import scipy.cluster.vq as scv
from sklearn.cluster import KMeans
from sklearn import metrics
from sklearn.metrics import pairwise_distances

## Create a function that takes hierarchically clustered data and produces a dictionary of K clusters. 
def clusterFromHc(hc_matrix, indexes, numObservations, k):


	# Create a dictionary to store each cluster.
	wardLinkage = hc_matrix
	listofClusters = {}
	n = numObservations
	
	for i in range(0,n-k):
		listofClusters[n+i] = []
		
		if ( not(listofClusters.has_key(wardLinkage[i][0])) and not(listofClusters.has_key(wardLinkage[i][1])) ):

			
			listofClusters[n+i].append(tuple((wardLinkage[i][0],indexes[int(wardLinkage[i][0])-1])))
			listofClusters[n+i].append(tuple((wardLinkage[i][1],indexes[int(wardLinkage[i][1])-1])))


		if ( listofClusters.has_key(wardLinkage[i][0]) and not(listofClusters.has_key(wardLinkage[i][1])) ):
			listofClusters[n+i].append(tuple((wardLinkage[i][1],indexes[int(wardLinkage[i][1])-1])))
			aux1 = listofClusters.get(wardLinkage[i][0])
			
			while (len(aux1) > 0):
				listofClusters[n+i].append(aux1.pop())
			
			listofClusters.pop(wardLinkage[i][0])

		
		
		if ( not(listofClusters.has_key(wardLinkage[i][0])) and (listofClusters.has_key(wardLinkage[i][1])) ):
			
			listofClusters[n+i].append(tuple((wardLinkage[i][0],indexes[int(wardLinkage[i][0])-1])))
			aux2 = listofClusters.get(wardLinkage[i][1])
			
			while (len(aux2) > 0):
				listofClusters[n+i].append(aux2.pop())

			listofClusters.pop(wardLinkage[i][1])

		if ( (listofClusters.has_key(wardLinkage[i][0])) and (listofClusters.has_key(wardLinkage[i][1])) ):
			
			aux1 = listofClusters.get(wardLinkage[i][0])
			
			while (len(aux1) > 0):
				listofClusters[n+i].append(aux1.pop())
			
			listofClusters.pop(wardLinkage[i][0])

			
			aux2 = listofClusters.get(wardLinkage[i][1])
			
			while (len(aux2) > 0):
				listofClusters[n+i].append(aux2.pop())

			listofClusters.pop(wardLinkage[i][1])

	keys = listofClusters.keys()
	numKeys = len(keys)
	newKeys = np.linspace(0,numKeys-1, num=k)
	
	
	for i in range(0,numKeys):
		listofClusters[newKeys[i]] = listofClusters.pop(keys[i])
	
	return listofClusters

## Implement a Psuedo-F statistic that takes a clustering object and returns fitness metric for determined k.
def pseudof(clusterObject, df, k):
	
		
	numObjects = df.shape[0]
	numFeatures = df.shape[1]
	
	dicClusters = clusterObject		

	centers = computeCentroids(clusterObject)

	# Center D is the center considering the initial dataset.				
	centerD = computeCenterofData(df)

	# Number of observations in each cluster. 
	nelemClusters = {}
	for key in clusterObject.keys():
		nelemClusters[key] = len(clusterObject[key])
	
	# Parameters of the PseudoF Index.
	NC = k


	# Constant: (n - NC)/(NC -1) .. out of the sum.
	const = float(float(numObjects - NC) / float(NC - 1))
	
	# Numerator: 
	num = 0
	for i in range(NC):
		num += nelemClusters[i] * ssd.euclidean(centers[i],centerD) * ssd.euclidean(centers[i],centerD)
	
	#print 'Num: {0}'.format(num)

	# Denominator
	den = 0
	for i in range(NC):
		for j in range(nelemClusters[i]):
			den += ssd.euclidean(dicClusters[i][j],centers[i]) * ssd.euclidean(dicClusters[i][j],centers[i])
	#print 'Den: {0}'.format(den)
	
	metric = const * (float(num) / float(den))
	#print 'Metric: {0}'.format(metric)
	
	#print 'K: {0} -- PseudoF Index: {1}'.format(k,metric)
	return metric

## Implement a Silhouette statistic that takes a clustering object and returns fitness metric for determined k.
def silhouette(df, labels):
	## Call the silhouette metric from metrics. SciLearn
	metric = metrics.silhouette_score(df, labels, metric='euclidean')
	return metric

## Create a pandas dataframe for the mrm .csv data.
def mrmData(path):

	df = pd.DataFrame.from_csv(path)
	
	# get data column names
	dataCols = []
	timePts = []
	for column in df:
	    
	    if re.search('data', column):
	        dataCols.append(column)
	        s = column.split(':') 		 
	        timePts.append(int(s[-1]))
	
	dfPhos = pd.DataFrame(columns=dataCols)
	 
	for i, row in df.iterrows():
	    rowVec = row[list(dataCols)]
	    dfPhos = dfPhos.append(rowVec)

	
	dfPhos.index = df['gene_site']
	
	return dfPhos

## Plot fitness metric for a list of metric. k_list is swept in [2,3,4 ..., kmax]
def plotMetric(k_list, metric1 , metric2, figure, title):

	
	# Plot Fitness Metric.
	plt.figure(figure)
	plt.axis([0 , k_list[-1]+1, -(abs(min(metric1)) + abs(min(metric2))), 1.5])
	plt.plot(k_list, metric1, 'bo')
	plt.hold('on')
	plt.plot(k_list, metric2, 'ro')
	plt.title(title)
	plt.ylabel('Indexes')
	plt.xlabel('K - Number of Clusters')
	plt.legend(['Normalized CH Index', 'Silhouette Index'])
	plt.savefig('Fitness Metric.png')
	plt.grid(True)
	plt.show()
	
	return

## For better comparison visualization, normVector takes the metric list swept in [2,3,...,kmax] and normalizes it in [0,1]
def normVector(pseudof_metric):
	
	max_value = max(pseudof_metric)
	norm_pseudof_metric = []
	for i in range(len(pseudof_metric)):
		norm_pseudof_metric.append(float(pseudof_metric[i]) / float(max_value))

	#print 'PseudoF norm metric: {0}'.format(norm_psudof_metric)
	
	return norm_pseudof_metric

## Ward linkage hierarchical clustering.
def wardLinkage(coocdf):


	# Adding zeros in the diagonal.
	for i in range(coocdf.shape[0]):
		for j in range(coocdf.shape[1]):
			if (i == j) and (coocdf.iloc[i,j] != 0):
				coocdf.iloc[i,j] = 0


	# Normalizing the co-occurrence matrix dataframe.
	cooc_norm = coocdf / max(coocdf.max())
	#print cooc_norm.iloc[0:5,0:5]

	# Distance matrix from co-occurrence matrix: dist = 1 - cooc.
	distdf = 1 - cooc_norm 
	#print distdf.iloc[0:5,0:5]

	for i in range(distdf.shape[0]):
		for j in range(distdf.shape[1]):
			if (i == j) and (distdf.iloc[i,j] != 0):
				distdf.iloc[i,j] = 0


	###
	# Co-occurrence matrix must to be changed to distance matrix to be passed as paramenter of linkage.
	###

	## Ward Linkage from Distance matrix.
	wlFromDistance = sch.linkage(distdf, method='ward', metric='euclidean')

	return wlFromDistance

## Create a cluster object from the data, observation indexes, and labels from the clustering method.
def createClusterObject(df, labels, clusterFromHc, switch):
	
	## dic: {0:([0 1 2 3 4 5 6],'ABCD'), ([0 1 2 3 4 5 6],'DCBA') .. 1: ... 2: ... k-1: ...}
	
	if switch == 'Part B':

		# Indexes of data
		indexes = df.index.values
		dic = {}
		numObservations = len(labels)
		
		for i in range(max(labels)+1):
			dic[i] = [] 

		
		# i: range 0 to 203
		for i in range(numObservations):
			#dic[labels[i]].append(tuple((df.iloc[i,:].tolist(),indexes[i])))
			dic[labels[i]].append(df.iloc[i,:].tolist())
		
		return dic
	
	elif switch == 'Part A':

		dic = {}
		for key in clusterFromHc.keys():
			dic[key] = []
			#print 'key: {0}'.format(key)
			for index in clusterFromHc[key]:
				#print 'obs: {0}'.format(index)
				dic[key].append(df.loc[index[1],:].tolist()) 

		
		return dic

## Compute centroids from ClusterObject and data.
def computeCentroids(clusterObject):
	
	keys = clusterObject.keys()
	numFeatures = len(clusterObject[np.random.choice(keys)][0])
	centroids = {}
	center = []
	
	aux = 0
	for key in keys:
		for y in range(numFeatures):
			for x in range(len(clusterObject[key])):
				aux += clusterObject[key][x][y]
			center.append(aux/len(clusterObject[key]))
			aux = 0
		centroids[key] = center
		center = []
	
	return centroids

## Compute the center of the entire dataset.
def computeCenterofData(df):

	numFeatures = df.shape[1]
	centerD = []
	[centerD.append(np.mean(df.iloc[:,i])) for i in range(numFeatures)]

	return centerD

## Test dataset.
def testDataset(i):
	a = pd.DataFrame(np.random.randn(50, 2))
	b = pd.DataFrame(50 + np.random.randn(50, 2))
	c = pd.DataFrame(100 + np.random.randn(50, 2))
	d = pd.DataFrame(150 + np.random.randn(50, 2))
	rand_df = pd.concat([a,b,c,d])
	rand_df = pd.DataFrame(rand_df.values, index= np.arange(200))

	cooc = {}
	for i in range(rand_df.shape[0]):
		cooc[i] = []
		for j in range(rand_df.shape[0]):
			cooc[i].append(ssd.euclidean(rand_df.iloc[i,:],rand_df.iloc[j,:]))
	cooc_df = pd.DataFrame(cooc)
	return rand_df, cooc_df

## Find cluster key given the value(observation).
def find_key(dic, value):
	
	keys = dic.keys()

	for key in keys:
		for cluster in dic[key]:
			if (cluster[0] == value):
				return key

## Creates label vector from the clusterFromHC. 
def labelsFromHC(clustersFromHc):
	
	numObservations = 0
	for key in clustersFromHc.keys():
		numObservations += len(clustersFromHc[key])
	
	labels = []

	for i in range(numObservations):
		labels.append(find_key(clustersFromHc,i))

	return labels
