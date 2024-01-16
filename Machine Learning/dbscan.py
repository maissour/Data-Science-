from sklearn.cluster import KMeans
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from numpy import unique, where


df= pd.read_csv('path/to/Mall_Customers.csv')

#Select the annual income and the spending score columns 
x = df.iloc[:, [3,4]].values

# WCSS = []
# for i in range(1,11):
#     kmeans_model = KMeans(n_clusters=i,init='k-means++') 
#     kmeans_model.fit(X)
#     WCSS.append(kmeans_model.inertia_)
# plt.plot(range(1,11),WCSS)


# kmeans = KMeans(n_clusters=5, init = 'k-means++')
# y_kmeans = kmeans.fit_predict(x)

# plt.scatter(x[y_kmeans==0,0],x[y_kmeans==0,1],s=100,c='red',label = 'cluster 1')
# plt.scatter(x[y_kmeans==1,0],x[y_kmeans==1,1],s=100,c='blue',label = 'cluster 2')
# plt.scatter(x[y_kmeans==2,0],x[y_kmeans==2,1],s=100,c='green',label = 'cluster 3')
# plt.scatter(x[y_kmeans==3,0],x[y_kmeans==3,1],s=100,c='violet',label = 'cluster 4')
# plt.scatter(x[y_kmeans==4,0],x[y_kmeans==4,1],s=100,c='brown',label = 'cluster 5')
# plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1])
from itertools import product

eps_values = np.arange(8,12.75,0.25) # eps values to be investigated
min_samples = np.arange(3,10) # min_samples values to be investigated
DBSCAN_params = list(product(eps_values, min_samples))
color=['red','blue','green' ,'violet','brown']

# define the model
model = DBSCAN(eps=9.25, min_samples=3)
# fit model and predict clusters
ypred = model.fit_predict(x)
# retrieve unique clusters
clusters = unique(ypred)
# create scatter plot for samples from each cluster
for cluster in clusters:
 	# get row indexes for samples with this cluster
 	row_ix = where(ypred == cluster)
 	# create scatter of these samples
 	plt.scatter(x[row_ix, 0], x[row_ix, 1])
# show the plot
plt.show()
