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
			if not re.match('__', method) and not re.match('_validation__', method):
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


	## The Det_Ratio index DRI
	def det_ratio(self):
		"""
		The determinant ratio index, a measure of connectedness
		"""
		#compute the attributes number and cluster number
		self.description = 'Determinant ratio, a measure of connectedness'
		attributes=len(self.dataMatrix[0])
		xData=self.dataMatrix
		wg=np.zeros((attributes,attributes))
		numCluster=max(self.classLabel)+1
		#compute cluster scatter matrix
		for i in range(numCluster):
			indices=[t for t, x in enumerate(self.classLabel) if x == i]
			clusterMember=self.dataMatrix[indices,:]
			xCluster=clusterMember
			#iterate through attributes
			for j in range(attributes):
				columnVec=clusterMember[:,j]
				columnCenter=np.mean(columnVec)
				#compute xk
				xCluster[:,j]=columnVec-columnCenter
			#add to wg
			wg=wg+np.dot(np.transpose(xCluster),xCluster)
		#compute data scatter matrix
		for i in range(attributes):
			columnVec=self.dataMatrix[:,i]
			columnCenter=np.mean(columnVec)
			#data scatter matrix
			xData[:,i]=columnVec-columnCenter

		t=np.dot(np.transpose(xData),xData)
		#compute the fitness
		self.validation = np.linalg.det(t)/np.linalg.det(wg)
		return self.validation

	def c_index(self):
		"""
		The C-Index, a measure of compactness
		"""
		self.description = 'The C-Index, a measure of cluster compactness'
		sw=0
		nw=0
		numCluster=max(self.classLabel)+1
		#iterate through all the clusters
		for i in range(numCluster):
			indices=[t for t, x in enumerate(self.classLabel) if x == i]
			clusterMember=self.dataMatrix[indices,:]
			#compute distance of every pair of points
			list_clusterDis=distance.pdist(clusterMember)
			sw=sw+sum(list_clusterDis)
			nw=nw+len(list_clusterDis)
		#compute the pairwise distance of the whole dataset
		list_dataDis=distance.pdist(self.dataMatrix)
		#compute smin
		sortedList=sorted(list_dataDis)
		smin=sum(sortedList[0:nw])
		#compute smax
		sortedList=sorted(list_dataDis,reverse=True)
		smax=sum(sortedList[0:nw])
		#compute the fitness
		self.validation = (sw-smin)/(smax-smin)
		return self.validation

	def g_plus_index(self):
		"""
		The G_plus index, the proportion of discordant pairs among all the pairs of distinct point, a measure of connectedness
		"""
		self.description = "The G_plus index, a measure of connectedness"
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
					#heter points larger than homo points
					if pairDis[i]>vecB[j]:
						sminus=sminus+1
				if vecB[i]==0 and vecB[j]>0:
					#heter points larger than homo points
					if pairDis[j]>vecB[i]:
						sminus=sminus+1
		#return fitness
		self.validation =  2*sminus/(numPair*(numPair-1))
		return self.validation

	def ksq_detw_index(self):
		"""
		The Ksq_DetW Index, a measure of connectedness
		"""
		self.description = "The Ksq_DetW index, a measure of connectedness"
		#compute the attributes number and cluster number
		attributes=len(self.dataMatrix[0])
		wg=np.zeros((attributes,attributes))
		numCluster=max(self.classLabel)+1
		#compute cluster scatter matrix
		for i in range(numCluster):
			indices=[t for t, x in enumerate(self.classLabel) if x == i]
			clusterMember=self.dataMatrix[indices,:]
			xCluster=clusterMember
			#iterate through attributes
			for j in range(attributes):
				columnVec=clusterMember[:,j]
				columnCenter=np.mean(columnVec)
				#compute xk
				xCluster[:,j]=columnVec-columnCenter
			#add to wg
			wg=wg+np.dot(np.transpose(xCluster),xCluster)
		#compute fitness
		self.validation = math.pow(numCluster,2)*np.linalg.det(wg)
		return self.validation

	def log_det_ratio(self):
		"""
		The log determinant ratio index, a measure of connectedness
		"""
		self.description = "The log determinant ratio index, a measure of connectedness"
		numObj=len(self.classLabel)
		self.validation = numObj*math.log(self.det_ratio())
		return self.validation

	def log_ss_ratio(self):
		"""
		The log ss ratio, a measure of connectedness
		"""
		self.description = "The log ss ratio, a measure of connectedness"
		bgss=0
		wgss=0
		numCluster=max(self.classLabel)+1
		#compute the dataset center
		dataCenter=np.mean(self.dataMatrix,0)
		#iterate through the cluster
		for i in range(numCluster):
			sumTemp=0
			indices=[t for t, x in enumerate(self.classLabel) if x == i]
			clusterMember=self.dataMatrix[indices,:]
			#compute the center of the cluster
			clusterCenter=np.mean(clusterMember,0)
			#add to bgss
			bgss=bgss+len(indices)*math.pow(distance.euclidean(clusterCenter, dataCenter),2)
			#iterate through all the members of the cluster
			for member in clusterMember:
				sumTemp=sumTemp+math.pow(distance.euclidean(member, clusterCenter),2)
			wgss=wgss+sumTemp
		#compute the fitness
		self.validation = math.log(bgss/wgss)
		return self.validation

	def McClain_Rao(self):
		"""
		The McClain-Rao Index, a measure of compactness
		"""
		self.description = "The McClain-Rao Index, a measure of compactness"
		sw=0
		sb=0
		nw=0
		numObj=len(self.classLabel)
		numCluster=max(self.classLabel)+1
		#iterate through all the clusters
		for i in range(numCluster):
			indices=[t for t, x in enumerate(self.classLabel) if x == i]
			clusterMember=self.dataMatrix[indices,:]
			#compute pairwise distance
			pairDis=distance.pdist(clusterMember)
			#add to sw and nw
			sw=sw+sum(pairDis)
			nw=nw+len(pairDis)
			#iterate the clusters again for between-cluster distance
			for j in range(numCluster):
				if j>i:
					indices2=[t for t, x in enumerate(self.classLabel) if x == j]
					clusterMember2=self.dataMatrix[indices2,:]
					betweenDis=distance.cdist(clusterMember,clusterMember2)
					#add to sb
					sb=sb+sum(list(itertools.chain(*betweenDis)))
		#compute nb
		nb=numObj*(numObj-1)/2-nw
		#compute fitness
		self.validation = nb*sw/(nw*sb)
		return self.validation

	def PBM_index(self):
		"""
		The PBM index, a measure of compactness
		"""
		self.description = "The PBM index, a measure of compactness"
		ew=0
		et=0
		list_centerDis=[]
		numCluster=max(self.classLabel)+1
		#compute the center of the dataset
		dataCenter=np.mean(self.dataMatrix,0)
		#iterate through the  clusters
		for i in range(numCluster):
			indices=[t for t, x in enumerate(self.classLabel) if x == i]
			clusterMember=self.dataMatrix[indices,:]
			#compute the center of the cluster
			clusterCenter=np.mean(clusterMember,0)
			#compute the center distance
			list_centerDis.append(distance.euclidean(dataCenter, clusterCenter))
			#iterate through the member of the  cluster
			for  member in clusterMember:
				ew=ew+distance.euclidean(member, clusterCenter)
				et=et+distance.euclidean(member, dataCenter)
		db=max(list_centerDis)
		#compute the fitness
		self.validation = math.pow(et*db/(numCluster*ew),2)
		return self.validation

	def point_biserial(self):
		"""
		The Point-Biserial index, a measure of connectedness
		"""
		self.description = "The Point-Biserial index, a measure of connectedness"
		sw=0
		sb=0
		nw=0
		numObj=len(self.classLabel)
		numCluster=max(self.classLabel)+1
		nt=numObj*(numObj-1)/2
		#iterate through all the clusters
		for i in range(numCluster):
			indices=[t for t, x in enumerate(self.classLabel) if x == i]
			clusterMember=self.dataMatrix[indices,:]
			#compute pairwise distance
			pairDis=distance.pdist(clusterMember)
			#add to sw and nw
			sw=sw+sum(pairDis)
			nw=nw+len(pairDis)
			#iterate the clusters again for between-cluster distance
			for j in range(numCluster):
				if j>i:
					indices2=[t for t, x in enumerate(self.classLabel) if x == j]
					clusterMember2=self.dataMatrix[indices2,:]
					betweenDis=distance.cdist(clusterMember,clusterMember2)
					#add to sb
					sb=sb+sum(list(itertools.chain(*betweenDis)))
		#compute nb
		nb=nt-nw
		#compute fitness
		self.validation = ((sw/nw-sb/nb)*math.sqrt(nw*nb))/nt
		return self.validation

	def Ratkowsky_Lance(self):
		"""
		The Ratkowsky-Lance index, a measure of compactness
		"""
		self.description = "The Ratkowsky-Lance index, a measure of compactness"
		list_divide=[]
		attributes=len(self.dataMatrix[0])
		numCluster=max(self.classLabel)+1
		#iterate through the attributes
		for i in range(attributes):
			bgssj=0
			tssj=0
			columnVec=self.dataMatrix[:,i]
			columnCenter=np.mean(columnVec)
			#compute bgssj
			for j in range(numCluster):
				indices=[t for t, x in enumerate(self.classLabel) if x == j]
				columnCluster=self.dataMatrix[indices,:]
				centerCluster=np.mean(columnCluster)
				bgssj=bgssj+len(indices)*math.pow(centerCluster-columnCenter,2)
			#iterate through the  members of the column
			for member in columnVec:
				tssj=tssj+math.pow(member-columnCenter,2)
			list_divide.append(bgssj/tssj)
		r=sum(list_divide)/attributes
		#compute the  fitness
		self.validation = math.sqrt(r/numCluster)
		return self.validation

	def Ray_Turi(self):
		"""
		The Ray-Turi index, a measure of compactness
		"""
		self.description = "The Ray-Turi index, a measure of compactness"
		wgss=0
		list_centers=[]
		numCluster=max(self.classLabel)+1
		numObj=len(self.classLabel)
		for i in range(numCluster):
			wgssk=0
			indices=[t for t, x in enumerate(self.classLabel) if x == i]
			clusterMember=self.dataMatrix[indices,:]
			#compute the center of the cluster
			clusterCenter=np.mean(clusterMember,0)
			list_centers.append(clusterCenter)
			#iterate through the  cluster members
			for member in clusterMember:
				wgssk=wgssk+math.pow(distance.euclidean(member, clusterCenter),2)
			#add to wgsss
			wgss=wgss+wgssk
		#compute the min center dis
		minDis=math.pow(min(distance.pdist(list_centers)),2)
		#compute the fitness
		self.validation = wgss/(numObj*minDis)
		return self.validation

	def Scott_Symons(self):
		"""
		The Scott-Symons index, a measure of connectedness
		"""
		self.description = "The Scott-Symons index, a measure of connectedness"
		fitness=0
		#compute the attributes number and cluster number
		attributes=len(self.dataMatrix[0])
		numCluster=max(self.classLabel)+1
		#compute cluster scatter matrix
		for i in range(numCluster):
			indices=[t for t, x in enumerate(self.classLabel) if x == i]
			nk=len(indices)
			clusterMember=self.dataMatrix[indices,:]
			xCluster=clusterMember
			#iterate through attributes
			for j in range(attributes):
				columnVec=clusterMember[:,j]
				columnCenter=np.mean(columnVec)
				#compute xk
				xCluster[:,j]=columnVec-columnCenter
			#compute wgk
			wgk=np.dot(np.transpose(xCluster),xCluster)
			if np.linalg.det(wgk/nk) != 0:
				fitness=fitness+nk*math.log(np.linalg.det(wgk/nk))
			else:
				warnings.warn('Cannot calculate Scott_Symons, due to an undefined value', UserWarning)
		#return fitness
		self.validation = fitness
		return self.validation

	def tau_index(self):
		"""
		The Tau index, a measure of compactness
		"""
		self.description = "The Tau index, a measure of compactness"
		#compute nb,nw,nt
		nw=0
		numObj=len(self.classLabel)
		numCluster=max(self.classLabel)+1
		nt=numObj*(numObj-1)/2
		for i in range(numCluster):
			indices=[t for t, x in enumerate(self.classLabel) if x == i]
			nk=len(indices)
			nw=nw+nk*(nk-1)/2
		nb=nt-nw
		#compute s+ and s-
		splus=0
		sminus=0
		pairDis=distance.pdist(self.dataMatrix)
		numPair=nt
		temp=np.zeros((len(self.classLabel),2))
		temp[:,0]=self.classLabel
		vecB=distance.pdist(temp)
		#iterate through all the pairwise comparisons
		for i in range(int(numPair-1)):
			for j in range(i+1,int(numPair)):
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
		self.validation = (splus-sminus)/math.sqrt(nb*nw*nt*(nt-1)/2)
		return self.validation

	def trace_w(self):
		"""
		The Trace_W index, a measure of connectedness
		"""
		self.description = "The Trace_W index, a measure of connectedness"
		wgss=0
		numCluster=max(self.classLabel)+1
		for i in range(numCluster):
			wgssk=0
			indices=[t for t, x in enumerate(self.classLabel) if x == i]
			clusterMember=self.dataMatrix[indices,:]
			#compute the center of the cluster
			clusterCenter=np.mean(clusterMember,0)
			#iterate through the  cluster members
			for member in clusterMember:
				wgssk=wgssk+math.pow(distance.euclidean(member, clusterCenter),2)
			#add to wgsss
			wgss=wgss+wgssk
		#return the fitness
		self.validation = wgss
		return self.validation

	def trace_wib(self):
		"""
		The Trace_WiB index, a measure of connectedness
		"""
		self.description = "The Trace_WiB index, a measure of connectedness"
		numCluster=max(self.classLabel)+1
		attributes=len(self.dataMatrix[0])
		b=np.zeros((numCluster,attributes))
		wg=np.zeros((attributes,attributes))
		#compute the data center
		dataCenter=np.mean(self.dataMatrix,0)
		#iterate all the clusters
		for i in range(numCluster):
			indices=[t for t, x in enumerate(self.classLabel) if x == i]
			clusterMember=self.dataMatrix[indices,:]
			xCluster=clusterMember
			#compute the center of the cluster
			clusterCenter=np.mean(clusterMember,0)
			b[i,:]=clusterCenter-dataCenter
			#compute wgk and wg
			for j in range(attributes):
				columnVec=clusterMember[:,j]
				columnCenter=np.mean(columnVec)
				#compute xk
				xCluster[:,j]=columnVec-columnCenter
			#compute wgk
			wg=wg+np.dot(np.transpose(xCluster),xCluster)
		#compute bg
		bg=np.dot(np.transpose(b),b)
		#compute fitness
		try:
			self.validation = np.trace(np.dot(np.linalg.inv(wg),bg))
		except np.linalg.linalg.LinAlgError:
			# Numpy will thrown an exception on singular matricies
			# If this happens, warn the user and return 0
			warnings.warn('Cannot calculate trace_wib, due to an undefined value', UserWarning)
			self.validation = 0
		return self.validation

	def Wemmert_Gancarski(self):
		"""
		The Wemmert-Gancarski index, the quotients of distances between the points and the barycenters of all clusters, a measure of compactness
		"""
		self.description = "The Wemmert-Gancarski index, a measure of compactness"
		sum=0
		list_centers=[]
		attributes=len(self.dataMatrix[0])
		numObj=len(self.classLabel)
		numCluster=max(self.classLabel)+1
		#compute all the centers
		for i in range(numCluster):
			indices=[t for t, x in enumerate(self.classLabel) if x == i]
			clusterMember=self.dataMatrix[indices,:]
			#compute the center of the cluster
			list_centers.append(np.mean(clusterMember,0))
		#iterate the clusters again for Rm
		for i in range(numCluster):
			sumRm=0
			indices=[t for t, x in enumerate(self.classLabel) if x == i]
			clusterMember=self.dataMatrix[indices,:]
			#compute the currrent center
			clusterCenter=np.mean(clusterMember,0)
			tempList=list_centers
			tempList = tempList[:i] + tempList[i+1 :]
			#iterate through the member and compute rm
			for member in clusterMember:
				#make it a 2d array
				memberArray=np.zeros((1,attributes))
				memberArray[0,:]=member
				#compute the pair wise distance
				list_dis=distance.cdist(memberArray,tempList)
				sumRm=sumRm+(distance.euclidean(member, clusterCenter))/min(min(list_dis))
			#compute the sum
			sum=sum+max([0,len(indices)-sumRm])
		#compute the fitness
		self.validation = sum/numObj
		return self.validation

	def root_mean_square(self):
		"""
		The Root-Mean-Square Standard Deviation (RMSSTD), the root mean square
		standard deviation of all variables within each cluster. A measure of
		connectedness.
		"""
		self.description = "The Root-Mean-Square Standard Deviation (RMSSTD), a measure of connectedness"
		numCluster=max(self.classLabel)+1
		attributes=len(self.dataMatrix[0])
		denominator=attributes*(len(self.dataMatrix)-numCluster)
		normSum=0
		#iterate through all the clusters
		for i in range(numCluster):
			indices=[t for t, x in enumerate(self.classLabel) if x == i]
			clusterMember=self.dataMatrix[indices,:]
			#compute the center of the cluster
			clusterCenter=np.mean(clusterMember,0)
			#compute the square error for every member in the cluster
			for member in clusterMember:
				normSum=normSum+distance.euclidean(member, clusterCenter)
		self.validation = math.sqrt(normSum/denominator)
		return self.validation
