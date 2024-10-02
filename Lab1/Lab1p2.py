import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.cluster.hierarchy import dendrogram, linkage #Imports support for making dendograms
from sklearn.cluster import AgglomerativeClustering #Imports support for making agglomerative clustering


customer_data = pd.read_csv('shopping_data.csv')

X = np.array([[5,3],
[10,15],
[15,12],
[24,10],
[30,30],
[85,70],
[71,80],
[60,78],
[70,55],
[80,91] ])

""" ### This is the code for the agglomerative clustering ###
cluster = AgglomerativeClustering(n_clusters=2, metric='euclidean', linkage='ward')
cluster.fit_predict(X) #This is the code that fits the data to the model

plt.scatter(X[:,0], X[:,1], c=cluster.labels_, cmap='rainbow') #This is the code that plots the data
plt.show() #This is the code that shows the plot

labels = range(1, 11) 
plt.figure(figsize=(10, 7))
plt.subplots_adjust(bottom=0.1)
plt.scatter(X[:,0],X[:,1], label='True Position')
for label, x, y in zip(labels, X[:, 0], X[:, 1]):
 plt.annotate(label,xy=(x, y),xytext=(-3, 3),textcoords='offset points', ha='right',va='bottom')
plt.show()

linked = linkage(X, 'single')
labelList = range(1, 11)
plt.figure(figsize=(10, 7))
dendrogram(linked,
 orientation='top',
 labels=labelList,
 distance_sort='descending',
 show_leaf_counts=True)
plt.show() """

#print(customer_data.shape)
#print(customer_data.head())

#Preprocessing the data to fit the model 
data = customer_data.iloc[:, 3:5].values

print(data.shape)
print(data)


### Dendrogram ###
cluster = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
cluster.fit_predict(data) #This is the code that fits the data to the model

#plt.scatter(data[:,0], data[:,1], c=cluster.labels_, cmap='rainbow') #This is the code that plots the data
#plt.show() #This is the code that shows the plot

linked = linkage(data, 'single')
labelList = range(1, 201)
plt.figure(figsize=(10, 7))
dendrogram(linked,
 orientation='top',
 labels=labelList,
 distance_sort='descending',
 show_leaf_counts=True)
plt.show()