#!/usr/bin/env python
# coding: utf-8

# In[3]:


from Precode import *
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd
import copy


# In[4]:


k1,i_point1,k2,i_point2 = initial_S1('5799')


# In[5]:


print(k1)
print(i_point1)
print(k2)
print(i_point2)


# In[6]:


print(data[0][0])


# In[7]:


def print_KMeans(data, k, points):
    plt.scatter([x[0] for x in data], [y[1] for y in data])
    for i in range(0,k):
        plt.scatter(points[i][0], points[i][1], label=f"c{i}")
    plt.legend()
    plt.show()


# In[8]:


print_KMeans(data, k1, i_point1)


# In[9]:


print_KMeans(data, k2, i_point2)


# In[10]:


print(k1)
print(i_point1)


# In[11]:


def KMeans(data, k, points_original):
    points = points_original.copy()
    iteration = 0
    centroids = {}
    #init the centroids
    for i in range(0,k):
        centroids[f"c{i}"] = points[i]
    #print(centroids)
    while True:
        iteration += 1
        classified_points = {}
        for key in centroids:
            classified_points[key] = list()
        #classify each point
        for i in range(0,len(data)):
            distances = []
            for key, val in centroids.items():
                diff = abs(np.subtract(data[i],val))
                distances.append(np.sqrt((diff[0] **2) + (diff[1] **2)))
            classified_points[f"c{np.argmin(distances)}"].append(data[i])
        #print(classified_points)
        #Calculate the new Mean
        new_centroids = {}
        for key, val in classified_points.items():
            new_centroids[key] = np.array([np.mean([x[0] for x in classified_points[key]]), np.mean([x[1] for x in classified_points[key]])])
#         print(centroids.values())
#         print(new_centroids.values())
        total = 0
        for key,v in centroids.items():
            if(centroids[key].tolist() == new_centroids[key].tolist()):
                total += 1
        if(total == k):
            print(f"This took {iteration} iterations.")
            return [x for x in new_centroids.values()]
        else:
            centroids = new_centroids


# In[12]:


news = KMeans(data, k1, i_point1)
print(news)
print_KMeans(data, k1, news)


# In[49]:


print_KMeans(data, k1, i_point1)


# In[50]:


print_KMeans(data, k2, i_point2)


# In[51]:


news = KMeans(data, k2, i_point2)
print(news)
print_KMeans(data, k2, news)


# In[54]:


def loss_fn(data, k, points):
    centroids = {}
    #init the centroids
    for i in range(0,k):
        centroids[f"c{i}"] = points[i]
    classified_points = {}
    for key in centroids:
            classified_points[key] = 0
    #classify each point
    for i in range(0,len(data)):
        distances = []
        for key, val in centroids.items():
            diff = abs(np.subtract(data[i],val))
            distances.append(np.sqrt((diff[0] **2) + (diff[1] **2)))
        classified_points[f"c{np.argmin(distances)}"] += (min(distances)) ** 2
    return sum(classified_points.values())


# In[55]:


loss_fn(data, k1, KMeans(data, k1, i_point1))


# In[56]:


loss_fn(data, k2, KMeans(data, k2, i_point2))

