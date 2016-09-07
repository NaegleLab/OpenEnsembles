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
import networkx as nx
from collections import defaultdict

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
		return self.labels

class co_occurrence_linkage:
	"""
	Returns a final solution to the ensemble that is the agglomerative clustering of the co_occurrence matrix 
	according to the linkage passed by linkage (default is Average). 

	"""
	def __init__(self, co_occ_object, threshold, linkage='average'):
		self.coMat = co_occ_object
		self.N = len(self.coMat.co_matrix)
		self.labels = np.empty(self.N)
		self.K= 0 #number of clusters made by the cut, to be replaced
		self.linkage = linkage
		self.threshold = threshold

	def finish(self):
		"""
		This is the function that is called ona  co_occurrence_linkage object that agglomeratively clusters the co_occurrence_matrix (self.coMat.co_matrix)
		According to self.linkage (linkage parameter set in initialization of object) with clusters equal to self.K (also set in intialization)
		"""
		#first get linkage, then cut
		lnk = self.coMat.link(linkage=self.linkage)
		labels = self.coMat.cut(lnk, self.threshold)
		self.K = len(np.unique(labels)) 
		self.labels = labels
		return self.labels

class graph_closure:
	"""
	Returns a final solution of the ensemble based on treating the co-occurrence matrix as a weighted graph whose 
	solution is found from identifying network components within the graph
	"""
	def __init__(self, co_occ_matrix, threshold, clique_size = 3):
		self.co_matrix = co_occ_matrix
		self.K= 0 #number of clusters made 
		self.N = len(co_occ_matrix)
		self.labels = np.empty(self.N)
		self.threshold = threshold
		self.coMat_binary = np.array(self.co_matrix >= threshold).astype(int)
		self.clique_size = clique_size

	def finish(self):
		""" Finishes the ensemble by taking a binary adjacency matrix, defined in initilization according to the threshold given
		and percolates the cliques"""

		# From ConradLee on GitHUB
		def get_percolated_cliques(G, k):
		    perc_graph = nx.Graph()
		    cliques = [frozenset(c) for c in nx.find_cliques(G) if len(c) >= k]
		    perc_graph.add_nodes_from(cliques)

		    # First index which nodes are in which cliques
		    membership_dict = defaultdict(list)
		    for clique in cliques:
		        for node in clique:
		            membership_dict[node].append(clique)

		    # For each clique, see which adjacent cliques percolate
		    for clique in cliques:
		        for adj_clique in get_adjacent_cliques(clique, membership_dict):
		            if len(clique.intersection(adj_clique)) >= (k - 1):
		                perc_graph.add_edge(clique, adj_clique)

		    # Connected components of clique graph with perc edges
		    # are the percolated cliques
		    for component in nx.connected_components(perc_graph):
		        yield(frozenset.union(*component))

		def get_adjacent_cliques(clique, membership_dict):
		    adjacent_cliques = set()
		    for n in clique:
		        for adj_clique in membership_dict[n]:
		            if clique != adj_clique:
		                adjacent_cliques.add(adj_clique)
		    return adjacent_cliques

		G = nx.from_numpy_matrix(self.coMat_binary)
		y = get_percolated_cliques(G, self.clique_size)
		z = list(y)
		clusterNum = 0
		while z:
		    l = list(z.pop())
		    self.labels[l] = int(clusterNum)
		    clusterNum+=1
		self.labels = self.labels.astype(int)

		return self.labels



