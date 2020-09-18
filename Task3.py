#!/usr/bin/env python
# coding: utf-8

# # Task 3 Predict The Optimum Number of Clusters And Represent It Visually 

# #### Intern Sumit Joshi 

# In[1]:


import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt 
import pandas as pd 
from sklearn import datasets
iris = datasets.load_iris()
df = pd.DataFrame(iris.data,columns=iris.feature_names)


# In[2]:


df.head()


# ##### Finding the optimal no of clusters for kmeans? How does one determine the value of k ?

# In[8]:


x = df.iloc[:,[0,1,2,3]].values
from sklearn.cluster import KMeans
wcss = []
for i in range(1,11):
    kmeans = KMeans(n_clusters = i,init = 'k-means++',
                   max_iter = 300, n_init = 10, random_state = 0)
    kmeans.fit(x)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title('The elbow method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')  #within cluster sum of squares 
plt.show()


# You can clearly see why it is called 'The elbow method' from the above graph, the optimum clusters is where the elbow occurs. This is when the within cluster sum of squares (WCSS) doesn't decrease significantly with every iteration.
# 
# From this we choose the number of clusters as ** '3**'."

# In[15]:


kmeans3= KMeans(n_clusters=3)
y_kmeans3 = kmeans3.fit_predict(x)
print(y_kmeans3)


# In[16]:


kmeans3.cluster_centers_


# In[19]:


#Visualizing Clustering 
plt.scatter(x[:,0],x[:,1],c=y_kmeans3, cmap='rainbow')


# In[ ]:




