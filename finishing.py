############################################################################
# OpenEnsemble Project													   #
# Implementation of the article Mixture Models for Ensemble CLustering by Topchy  
############################################################################
import numpy as np
import pandas as pd
import pprint as pp
import copy
import matplotlib.pyplot as plt
import sklearn.cluster
import sys
import os

class mixture_model:
	"""
	Implementation of the article Mixture Models for Ensemble CLustering
	Topchy, Jain, and Punch, "A mixture model for clustering ensembles Proc. SIAM Int. Conf. Data Mining (2004)"
	"""

	def __init__(self, parg, N, nEnsCluster=2, iterations=10):
		
		self.parg = parg #list of lists of solutions
		self.N = N# number of data points
		self.nEnsCluster = nEnsCluster #number of clusters to make from ensemble
		self.iterations = iterations

		self.y = self.gatherPartitions()
		self.y
		self.K = self.genKj()
		self.alpha, self.v, self.ExpZ = self.initParameters()
		self.labels = []
		self.piFinishing = {}
		

	def gatherPartitions(self):
		'''
		Returns the y vector.
		parg: list of H-labeling solutions
		nElem: number of features/objects
		'''
		H = len(self.parg)
		listParts = np.concatenate(self.parg).reshape(H,self.N)
		#print listParts[:,0]

		y = [] 
		[y.append(listParts[:,i]) for i in range(self.N)]

		y = pd.DataFrame(y, columns= np.arange(H))
		y.index.name = 'objs'
		y.columns.name = 'partition'
		return y
	
	def genKj(self):
	    '''
	    Generates the K(j) H-array that contains the tuples of unique 
	    clusters of each j-th partition, eg: K = [(X,Y), (A,B)] 
	    '''
	    #K = np.zeros(y.shape[1], dtype= int)
	    K = []
	    aux = []
	    for i in range(self.y.shape[1]):
	        if 'NaN' in np.unique(self.y.iloc[:,i].values):
	            aux = copy.copy(self.y.iloc[:,i].values)
	            aux = [x for x in aux if x != 'NaN']
	            K.append(aux)
	        else:
	            K.append(tuple(np.unique(self.y.iloc[:,i].values)))
	    return K

	def initParameters(self):
	    '''
	    The function initializes the parameters of the mixture model.
	    '''    
	    def initAlpha():
	        return np.ones(self.nEnsCluster) / self.nEnsCluster
	        
	    def initV():
	        v = []
	        [v.append([]) for j in range(self.y.shape[1])]
	        
	        #[v[j].append(list(np.ones(len(self.K[j])) / len(self.K[j]))) for j in range(self.y.shape[1]) for m in range(self.nEnsCluster)]
	        for j in range(self.y.shape[1]):
	        	for m in range(self.nEnsCluster):
	        		aux = abs(np.random.randn(len(self.K[j])))
	        		v[j].append( aux / sum(aux) )
	        
	        return v
	    
	    def initExpZ():
	        return np.zeros(self.y.shape[0] * self.nEnsCluster).reshape(self.y.shape[0],self.nEnsCluster)
	    
	    alpha = initAlpha()
	    v = initV()
	    ExpZ = initExpZ()
	    return alpha, v, ExpZ


	def expectation(self):
	    '''
	    Compute the Expectation (ExpZ) according to parameters.
	    Obs: y(N,H) Kj(H) alpha(M) v(H,M,K(j)) ExpZ(N,M)
	    '''
	    def sigma(a,b):
	    	return 1 if a == b else 0

	    M = self.ExpZ.shape[1]
	    nElem = self.y.shape[0]
	    
	    
	    for m in range(M):
	        for i in range(nElem):
	            
	            prod1 = 1
	            num = 0
	            for j in range(self.y.shape[1]):
	                ix1 = 0
	                for k in self.K[j]:
	                    prod1 *= ( self.v[j][m][ix1] ** sigma(self.y.iloc[i][j],k) )
	                    ix1 += 1
	            num += self.alpha[m] * prod1
	            
	            den = 0
	            for n in range(M):
	            
	                prod2 = 1
	                for j2 in range(self.y.shape[1]):
	                    ix2 = 0
	                    for k in self.K[j2]:
	                        prod2 *= ( self.v[j2][n][ix2] ** sigma(self.y.iloc[i][j2],k) )
	                        ix2 += 1
	                den += self.alpha[n] * prod2
	            
	            
	            self.ExpZ[i][m] = float(num) / float(den)
	    
	    return self.ExpZ


	def maximization(self):
	    '''
	    Update the parameters taking into account the ExpZ computed in the 
	    Expectation (ExpZ) step.
	    Obs: y(N,H) Kj(H) alpha(M) v(H,M,K(j)) ExpZ(N,M)
	    '''
	    def vecSigma(vec, k):
		    '''
		    Compare i-th elements of vector to k assigining to 
		    a vector 1 if i-th == k, 0 otherwise. 

		    '''
		    aux = []
		    for i in vec:
		        if i == k:
		            aux.append(1)
		        else:
		            aux.append(0)
		    return np.asarray(aux)


	    def updateAlpha():
	        for m in range(self.alpha.shape[0]):
	            self.alpha[m] = float(sum(self.ExpZ[:,m])) / float(sum(sum(self.ExpZ)))
	        return self.alpha
	    
	    def updateV():
	        
	        for j in range(self.y.shape[1]):
	            for m in range(self.alpha.shape[0]):
	                ix = 0
	                for k in self.K[j]:
	                    num = sum(vecSigma(self.y.iloc[:,j],k) * np.array(self.ExpZ[:,m]))
	                    den = 0
	                    for kx in self.K[j]:
	                        den += sum(vecSigma(self.y.iloc[:,j],kx) * np.asarray(self.ExpZ[:,m]))
	                    self.v[j][m][ix] = float(num) / float(den)
	                    ix += 1
	                            
	        return self.v
	                
	    self.alpha = updateAlpha()
	    self.v = updateV()
	    return self.alpha, self.v


	def emProcess(self):
		
		def piConsensus():
		    '''
		    The function outputs the final ensemble solution based on ExpZ values.
		    '''
		    maxExpZValues = {}
		    piFinishing = {}
		    labels = []
		    for i in range(self.ExpZ.shape[0]):
		        maxExpZValues[i] = max(self.ExpZ[i,:]) 
		        piFinishing[i] = []
		        
		        for j in range(self.ExpZ.shape[1]):
		            if (maxExpZValues[i] == self.ExpZ[i,j]):
		                piFinishing[i].append(j + 1)
		                labels.append(j+1)
		    
		    # choose randomly in the case of same values of ExpZ[i,:]          
		    #[piFinishing[i].delete(random.choice(piFinishing[i])) for i in piFinishing.keys() if (len(piFinishing[i]) > 1)]             
		    return piFinishing, labels
		
		i = 0
		while(i<self.iterations):
			self.ExpZ = self.expectation()
			self.alpha, self.v = self.maximization()
			i += 1

		piFinishing, labels = piConsensus()
		self.piFinishing = piFinishing
		self.labels = np.asarray(labels)
		#	return piFinishing, labels
		return labels

class co_occurrence_linkage:
	"""
	Returns a final solution to the ensemble that is the agglomerative clustering of the co_occurrence matrix 
	according to the linkage passed by linkage (default is Ward). K=2 is also default.

	"""
	def __init__(self, cObj, K=2, linkage='Ward'):
		self.cObj = cObj
		self.coMat = cObj.co_occurrence_matrix()
		self.K= K #number of clusters to make from ensemble
		self.labels = []
		self.linkage = linkage

	#def link(self):
	"""
	This is the function that is called ona  co_occurrence_linkage object that agglomeratively clusters the co_occurrence_matrix (self.coMat.co_matrix)
	According to self.linkage (linkage parameter set in initialization of object) with clusters equal to self.K (also set in intialization)
	"""



