import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import datetime
import operator
import math
import random

import numpy as np
from scipy.linalg import norm
from scipy.spatial.distance import cdist
from collections import defaultdict


from sklearn import datasets
from sklearn.datasets import make_blobs

# ignoring warnings
import warnings
warnings.simplefilter("ignore")

#import os, cv2, json
np.random.seed(42)

iris=datasets.load_iris()

X=pd.DataFrame(iris.data,columns=['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width'])
y=pd.DataFrame(iris.target,columns=['Classes'])

X=X.values
y=y.values
data=X

def euclidean_distance(point1,point2):
    dis=0
    for i in range(len(point1)):
        dis+=(point1[i]-point2[i])**2
    return dis**0.5
# utility values
m=2
n=len(data)
c=3
p=len(data[0])
max_iter=100
def initialize_membership_matrix(n,c):
    member_mat=list()
    for i in range(n):
        random_list=[random.random() for x in range(c)]
        summation=sum(random_list)
        for i in range(len(random_list)):
            random_list[i]=random_list[i]/summation
        member_mat.append(random_list)
    return member_mat

def update_centroid(member_mat):
    centroids={}

    for j in range(c):
        temp=[]
        for k in range(p):
            
            add=0
            for i in range(n):
                add+=member_mat[i][j]**m
            x=0
            for i in range(n):
                x+=(member_mat[i][j]**m)*(data[i][k])
            val=x/add
            temp.append(val)
        centroids[j]=temp
    return centroids


def update_membership_matrix(member_mat,centroids):
    ratio=float(2/(m-1))

    for i in range(n):
        distances=list()
        for j in range(c):
             distances.append(euclidean_distance(data[i],centroids[j]))
        for j in range(c):
            den = sum([math.pow(float(distances[j]/distances[q]), ratio) for q in range(c)])
            member_mat[i][j] = float(1/den) 
           
            
    return member_mat
        
                
    
def find_cluster(member_mat):
    clusters=list()
    for i in range(n):
        max_val, idx = max((val, idx) for (idx, val) in enumerate(member_mat[i]))
        clusters.append(idx)
    return clusters

def check(old_member_mat,member_mat):
    diff=0
    for i in range(n):
        for j in range(c):
            diff+=old_member_mat[i][j]-member_mat[i][j]
    if(diff<0.01):
        return True
    return False

def fuzzy_c_mean():
    
    member_mat=initialize_membership_matrix(n,c)
    for i in range(max_iter):
        centroids=update_centroid(member_mat)
        old_member_mat=member_mat
        member_mat=update_membership_matrix(member_mat,centroids)
        cluster=find_cluster(member_mat)
        if(check(old_member_mat,member_mat))<0.01:
            print(i)
            break 
    return cluster,centroids

clusters,centroids=fuzzy_c_mean()
print("Final Centroid points are:")
print(centroids)


# labeling the clusters
def label_clusters(clusters):
    z=0
    o=0
    t=0
    dict=defaultdict(int)


    for i in range(50):
        if(clusters[i]==0):
            z=z+1
        elif(clusters[i]==1):
            o=o+1
        else:
            t=t+1
    dict[z]=0
    dict[o]=1
    dict[t]=2
    classes=[]
    fin1=max(z,max(o,t))
    
    classes.append(dict[fin1])
    z=0
    o=0
    t=0
    for i in range(50,100):
        if(clusters[i]==0):
            z=z+1
        elif(clusters[i]==1):
            o=o+1
        else:
            t=t+1
    dict[z]=0
    dict[o]=1
    dict[t]=2

    fin1=max(z,max(o,t))
    
    classes.append(dict[fin1])
    z=0
    o=0
    t=0
    for i in range(100,150):
        if(clusters[i]==0):
            z=z+1
        elif(clusters[i]==1):
            o=o+1
        else:
            t=t+1
    dict[z]=0
    dict[o]=1
    dict[t]=2
    fin1=max(z,max(o,t))
   
    classes.append(dict[fin1])
    
    return classes

              
   
plt.scatter(X[:,0],X[:,1],colors="r" )  
plt.axis('equal')                                                                 
plt.xlabel('Sepal Length', fontsize=16)                                                 
plt.ylabel('Sepal Width', fontsize=16)                                                 
plt.title('Initial Random Clusters(Sepal)', fontsize=22)                                            
plt.grid()                                                                        
plt.show()


    

