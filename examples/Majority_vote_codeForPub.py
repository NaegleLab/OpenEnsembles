#Reproduce Ana Fred's majority voting solution using OpenEnsembles
import pandas as pd 
from sklearn import datasets
import openensembles as oe

#Set up a dataset and put in pandas DataFrame.
x, y = datasets.make_blobs(n_samples=250, centers=[(0,0), (0, 10)], cluster_std=1)
df = pd.DataFrame(x) 

#instantiate the oe data object
dataObj = oe.data(df, [1,2])

#instantiate an oe clustering object
c = oe.cluster(dataObj) 

#Use a 
c_MV_arr = []
val_arr = []
for i in range(0,19):
    name = 'kmeans_' + str(i) #to append a new solution, it must have a name (dictionary key) that is unique
    c.cluster('parent', 'kmeans', name, K=16, init = 'random', n_init = 1) #c.cluster will eventually become numIterations long
    c_MV_arr.append(c.finish_majority_vote(threshold=0.5)) # calculate a new majority vote solution each time it has one more iteration
    #calculate the silhouette validation metric for each majority vote solution
    v = oe.validation(dataObj, c_MV_arr[i]) #instantiate with the majority vote cluster object
    output_name = v.calculate('silhouette', 'majority_vote', 'parent')
    val_arr.append(v.validation[output_name])

#calculate the co-occurrence matrix
coMat = c.co_occurrence_matrix()