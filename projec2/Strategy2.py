#!/usr/bin/env python
# coding: utf-8

# In[16]:


from Precode2 import *
import numpy as np

get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
import pandas as pd


# In[17]:


k1,i_point1,k2,i_point2 = initial_S2('5799')


# In[18]:


print(k1)
print(i_point1)
print(k2)
print(i_point2)


# In[19]:


class KMeans2(object):
    def __init__(self,  k: int,  points: list, data: list):
        self.clusters = None
        self.data = data
        self.k = k
        self.loss = None
        self.points = [points]

    def add_point(self, indices):
        max_distance = 0

        for i in range(len(self.data)):
            if i in indices.keys():
                continue
            running_distance = 0
        
            for j in range(len(self.points)):
                running_distance += np.linalg.norm(abs(self.points[j] - self.data[i]))

            if running_distance > max_distance:
                max_distance = running_distance
                idx = i
        
        indices[idx] = 1
        self.points.append(self.data[idx])
    
    def calc_initial_centers(self):
        indices = {}
        for i in range(self.k - 1):
            self.add_point(indices)

    def calc_k_means(self):
        changed = True
        while changed:
            self.clusters = {}

            for i in range(1, self.k + 1):
                self.clusters[i] = []

            for v in data:
                distance = float('inf')
                current_cluster = 0

                for i in range(len(self.points)):
                    dist = np.linalg.norm(self.points[i] - v)
                    if dist < distance:
                        distance = dist
                        current_cluster = i + 1
                self.clusters[current_cluster].append(v.tolist())

            # move mu for each cluster
            for i in range(len(self.points)):
                mu = np.mean(np.array(self.clusters[i + 1]), axis=0)
                if np.array_equal(mu, self.points[i]):
                    changed = False
                else:
                    changed = True
                    self.points[i] = mu

    def calc_object_function(self):
        summation = 0
        for i in range(self.k):
            for j in range(len(self.clusters[i+1])):
                summation += np.linalg.norm(np.array(self.clusters[i+1][j]) - np.array(self.points[i])) ** 2
        self.loss = summation

    def plot(self): 
        for k in KMS.clusters:
            for point in KMS.clusters[k]:
                plt.plot(point[0],point[1], 'o', label="Cluster='{0}'".format(k) )
            plt.plot(KMS.points[k-1][0],KMS.points[k-1][1], 'o', markersize=15, markeredgewidth=2.0, mec= 'k', label="Cluster='{0}'".format(k) )
        plt.show()

    def display_output(self):
        print('After KMeans Algorithm: \n', pd.DataFrame(KMS.points, columns=["X", "Y"]), '\n')
        print('Loss: ', KMS.loss)

class Points2(object):
    def get_random_points(k: int, data: list):
        indices = np.random.choice(data.shape[0], 1, replace=False)
        return data[indices]


# In[20]:


data = np.load('AllSamples.npy')

k1, i_point1, k2, i_point2 = initial_S2('5799') 

KMS = KMeans2(k1, i_point1, data)
KMS.calc_initial_centers()
KMS.calc_k_means()
KMS.calc_object_function()
KMS.display_output()
KMS.plot()


# In[21]:


KMS = KMeans2(k2, i_point2, data)
KMS.calc_initial_centers()
KMS.calc_k_means()
KMS.calc_object_function()
KMS.display_output()
KMS.plot()

