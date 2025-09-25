# -*- coding: utf-8 -*-
"""
Created on Wed Sep 24 20:22:09 2025

@author: mario
"""

# K-means

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.cluster import KMeans
from kneed import KneeLocator

#Read dataset
dataset = pd.read_csv("Mall_Customers.csv")

# Get all raws with the Annual Income and Spending Score
# Get the characteristics matrix
Annual_income_column = 3
Spending_score_column = 4
X = dataset.iloc[:,[Annual_income_column, Spending_score_column]].values


# Apply elbow method to find the optimal number of clusters (k) in k-means clustering
# Calculate Within Cluster Sum of Squares

wcss = []

min_k = 1
max_k = 10

for i in range(min_k, max_k + 1):
    kmeans = KMeans(n_clusters = i, init = 'k-means++',n_init = 10, max_iter=300, random_state=0)
    
    kmeans.fit(X)
    
    wcss.append(kmeans.inertia_)
    
plt.plot(range(min_k, max_k+1), wcss)
plt.title("Elbow method")
plt.xlabel("Cluters number")
plt.ylabel("WCSS(k)")
plt.show()    
    
#Once wcss is calculated, apply kneedle algorithm to get the optimal k
kl = KneeLocator(range(min_k, max_k+1), wcss, curve='convex', direction='decreasing')
optimal_k = kl.elbow        
    
#Apply k-means method to segment the data set with the optimal k
kmeans = KMeans(n_clusters = optimal_k, init = 'k-means++',n_init = 10, max_iter=300, random_state=0)
    
y_kmeans = kmeans.fit_predict(X)    
    
    
# Plot clusters
cluster_point_size = 80
barycenter_point_size = 300

plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = cluster_point_size, c = 'red', label = 'Thrifty')    

plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = cluster_point_size, c = 'blue', label = 'Standard')

plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = cluster_point_size, c = 'green', label = 'Objetive')

plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = cluster_point_size, c = 'cyan', label = 'Careless')

plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = cluster_point_size, c = 'magenta', label = 'Conservatives')

plt.scatter(kmeans.cluster_centers_[:,0], kmeans.cluster_centers_[:,1], s = barycenter_point_size, c = "yellow", label = "Barycenters")
    
plt.title('Client cluster')    


plt.xlabel('Annual Income $k')
plt.ylabel('Spending score (1-100')
plt.legend()
plt.show()



    
    
    
    
    