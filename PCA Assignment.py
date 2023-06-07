#!/usr/bin/env python
# coding: utf-8

# Perform Principal component analysis and perform clustering using first 
# 3 principal component scores (both heirarchial and k mean clustering(scree plot or elbow curve) and obtain 
# optimum number of clusters and check whether we have obtained same number of clusters with the original data 
# (class column we have ignored at the begining who shows it has 3 clusters)df

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('wine.csv')
df


# In[3]:


df.shape


# In[4]:


df.head()


# In[5]:


from sklearn.decomposition import PCA


# In[6]:


pca=PCA()
df1=pd.DataFrame(pca.fit_transform(df))
df1.columns=list(df)
df1


# In[7]:


t1 = pca.explained_variance_ratio_


# In[8]:


t1[0]


# In[9]:


t1[1]


# In[10]:


t1[2]


# In[38]:


X=df1.iloc[:,:3]
X


# In[39]:


get_ipython().run_line_magic('matplotlib', 'qt')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(16,9)


# In[45]:


from mpl_toolkits.mplot3d import Axes3D

fig1=plt.figure()
ax=Axes3D(fig1)
ax.scatter(X['Type'],X['Alcohol'],X['Malic'])
plt.show()


# In[41]:


from sklearn.cluster import AgglomerativeClustering


# In[49]:


cluster1=AgglomerativeClustering(n_clusters=6,affinity='euclidean',linkage='complete')
Y1=cluster1.fit_predict(X)
Y_new1=pd.DataFrame(Y1)
Y_new1[0].value_counts()


# In[50]:


plt.figure(figsize=(10,7))
plt.scatter(X['Type'],X['Alcohol'],c=cluster1.labels_,cmap='rainbow')


# In[52]:


cluster2=AgglomerativeClustering(n_clusters=6,affinity='euclidean',linkage='complete')
Y2=cluster2.fit_predict(X)
Y_new2=pd.DataFrame(Y2)
Y_new2[0].value_counts()


# In[53]:


plt.figure(figsize=(10,7))
plt.scatter(X['Type'],X['Malic'],c=cluster1.labels_,cmap='rainbow')


# In[56]:


Y_clust1=pd.DataFrame(Y1)
Y_clust1[0].value_counts()


# In[57]:


Y_clust2=pd.DataFrame(Y2)
Y_clust2[0].value_counts()


# In[58]:


from sklearn.cluster import KMeans


# In[59]:


l1=[]
for i in range(1,17):
    kmeans=KMeans(n_clusters=i)
    kmeans=kmeans.fit(X)
    l1.append(kmeans.inertia_)
print(l1)


# In[60]:


import matplotlib.pyplot as plt

plt.scatter(range(1,17),l1)
plt.plot(range(1,17),l1,color='red')
plt.show()


# In[61]:


from sklearn.preprocessing import StandardScaler

SS=StandardScaler()
SS_X=SS.fit_transform(X)
SS_X


# In[62]:


from sklearn.cluster import DBSCAN

dbscan=DBSCAN(eps=1,min_samples=3)
dbscan.fit(SS_X)


# In[63]:


Y1=dbscan.labels_
Y_new1=pd.DataFrame(Y1)
Y_new1[0].value_counts()


# In[72]:


df1=pd.concat([X,Y_new1],axis=1)
df1


# In[73]:


df1[df1[0]==-1]


# In[74]:


df_new1=df1[df1[0]!=-1]
df_new1.shape


# In[75]:


Y2=dbscan.labels_
Y_new2=pd.DataFrame(Y2)
Y_new2[0].value_counts()


# In[76]:


df2=pd.concat([X,Y_new1],axis=1)
df2


# In[77]:


df2[df2[0]==-1]


# In[78]:


df_new2=df2[df2[0]!=-1]
df_new2.shape


# In[ ]:




