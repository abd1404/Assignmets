#!/usr/bin/env python
# coding: utf-8

# 1) Perform Clustering(Hierarchical, Kmeans & DBSCAN) for the crime data and identify the number of clusters formed and draw inferences.
# 
# Data Description:
# Murder -- Muder rates in different places of United States
# Assualt- Assualt rate in different places of United States
# UrbanPop - urban population in different places of United States
# Rape - Rape rate in different places of United States

# In[1]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[33]:


df=pd.read_csv('crime_data.csv',delimiter=',')
df


# In[34]:


df.shape


# In[35]:


df.head()


# In[36]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()
df['Unnamed: 0']=LE.fit_transform(df['Unnamed: 0'])


# In[37]:


X=df.iloc[:,0:5].values
X.shape


# In[38]:


get_ipython().run_line_magic('matplotlib', 'qt')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(16,9)


# In[41]:


from mpl_toolkits.mplot3d import Axes3D

fig1=plt.figure()
ax=Axes3D(fig1)
ax.scatter(X[:,0],X[:,1],X[:,2])
plt.show()


# In[42]:


fig2=plt.figure()
ax=Axes3D(fig2)
ax.scatter(X[:,0],X[:,3],X[:,4])
plt.show()


# In[27]:


from sklearn.cluster import AgglomerativeClustering


# In[102]:


cluster1=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='single')
Y1=cluster1.fit_predict(X)
Y_new1=pd.DataFrame(Y1)
Y_new1[0].value_counts()


# In[103]:


plt.figure(figsize=(10,7))
plt.scatter(X[:,0],X[:,1],c=cluster.labels_,cmap='rainbow')


# In[104]:


cluster2=AgglomerativeClustering(n_clusters=6,affinity='euclidean',linkage='ward')
Y2=cluster2.fit_predict(X)
Y_new2=pd.DataFrame(Y2)
Y_new2[0].value_counts()


# In[105]:


plt.figure(figsize=(10,7))
plt.scatter(X[:,0],X[:,1],c=cluster2.labels_,cmap='rainbow')


# In[106]:


cluster3=AgglomerativeClustering(n_clusters=6,affinity='euclidean',linkage='average')
Y3=cluster3.fit_predict(X)
Y_new3=pd.DataFrame(Y3)
Y_new3[0].value_counts()


# In[107]:


plt.figure(figsize=(10,7))
plt.scatter(X[:,0],X[:,2],c=cluster3.labels_,cmap='rainbow')


# In[108]:


cluster4=AgglomerativeClustering(n_clusters=7,affinity='euclidean',linkage='complete')
Y4=cluster4.fit_predict(X)
Y_new4=pd.DataFrame(Y4)
Y_new4[0].value_counts()


# In[109]:


plt.figure(figsize=(10,7))
plt.scatter(X[:,0],X[:,2],c=cluster4.labels_,cmap='rainbow')


# In[110]:


cluster5=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='complete')
Y5=cluster5.fit_predict(X)
Y_new5=pd.DataFrame(Y5)
Y_new5[0].value_counts()


# In[111]:


plt.figure(figsize=(10,7))
plt.scatter(X[:,0],X[:,3],c=cluster5.labels_,cmap='rainbow')


# In[112]:


cluster6=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='single')
Y6=cluster6.fit_predict(X)
Y_new6=pd.DataFrame(Y6)
Y_new6[0].value_counts()


# In[113]:


plt.figure(figsize=(10,7))
plt.scatter(X[:,0],X[:,3],c=cluster6.labels_,cmap='rainbow')


# In[114]:


cluster7=AgglomerativeClustering(n_clusters=6,affinity='euclidean',linkage='ward')
Y7=cluster7.fit_predict(X)
Y_new7=pd.DataFrame(Y7)
Y_new7[0].value_counts()


# In[115]:


plt.figure(figsize=(10,7))
plt.scatter(X[:,0],X[:,4],c=cluster7.labels_,cmap='rainbow')


# In[116]:


cluster8=AgglomerativeClustering(n_clusters=7,affinity='euclidean',linkage='average')
Y8=cluster8.fit_predict(X)
Y_new8=pd.DataFrame(Y8)
Y_new8[0].value_counts()


# In[117]:


plt.figure(figsize=(10,7))
plt.scatter(X[:,0],X[:,4],c=cluster7.labels_,cmap='rainbow')


# In[118]:


Y_clust1=pd.DataFrame(Y1)
Y_clust1[0].value_counts()


# In[119]:


Y_clust2=pd.DataFrame(Y2)
Y_clust2[0].value_counts()


# In[120]:


Y_clust3=pd.DataFrame(Y3)
Y_clust3[0].value_counts()


# In[121]:


Y_clust3=pd.DataFrame(Y3)
Y_clust3[0].value_counts()


# In[122]:


Y_clust4=pd.DataFrame(Y4)
Y_clust4[0].value_counts()


# In[123]:


Y_clust5=pd.DataFrame(Y5)
Y_clust5[0].value_counts()


# In[124]:


Y_clust6=pd.DataFrame(Y6)
Y_clust6[0].value_counts()


# In[125]:


Y_clust7=pd.DataFrame(Y7)
Y_clust7[0].value_counts()


# In[126]:


Y_clust8=pd.DataFrame(Y8)
Y_clust8[0].value_counts()


# In[127]:


from sklearn.cluster import KMeans


# In[128]:


l1=[]
for i in range(1,15):
    kmeans=KMeans(n_clusters=i)
    kmeans=kmeans.fit(X)
    l1.append(kmeans.inertia_)
print(l1)


# In[129]:


import matplotlib.pyplot as plt

plt.scatter(range(1,15),l1)
plt.plot(range(1,15),l1,color='red')
plt.show()


# In[130]:


from sklearn.preprocessing import StandardScaler

SS=StandardScaler()
SS_X=SS.fit_transform(X)
SS_X


# In[161]:


from sklearn.cluster import DBSCAN

dbscan=DBSCAN(eps=1,min_samples=3)
dbscan.fit(SS_X)


# In[162]:


Y1=dbscan.labels_
Y_new1=pd.DataFrame(Y1)
Y_new1[0].value_counts()


# In[163]:


df1=pd.concat([df,Y_new1],axis=1)
df1


# In[164]:


df1[df1[0]==-1]


# In[165]:


df_new1=df1[df1[0]!=-1]
df_new1.shape


# In[166]:


Y2=dbscan.labels_
Y_new2=pd.DataFrame(Y2)
Y_new2[0].value_counts()


# In[167]:


df2=pd.concat([df,Y_new2],axis=1)
df2


# In[168]:


df2[df2[0]==-1]


# In[169]:


df_new2=df2[df2[0]!=-1]
df_new2.shape


# In[170]:


Y3=dbscan.labels_
Y_new3=pd.DataFrame(Y3)
Y_new3[0].value_counts()


# In[171]:


df3=pd.concat([df,Y_new3],axis=1)
df3


# In[172]:


df3[df3[0]==-1]


# In[173]:


df_new3=df3[df3[0]!=-1]
df_new3.shape


# In[174]:


Y4=dbscan.labels_
Y_new4=pd.DataFrame(Y4)
Y_new4[0].value_counts()


# In[175]:


df4=pd.concat([df,Y_new4],axis=1)
df4


# In[176]:


df4[df4[0]==-1]


# In[177]:


df_new4=df4[df4[0]!=-1]
df_new4.shape


# In[178]:


Y5=dbscan.labels_
Y_new5=pd.DataFrame(Y5)
Y_new5[0].value_counts()


# In[179]:


df5=pd.concat([df,Y_new5],axis=1)
df5


# In[180]:


df5[df5[0]==-1]


# In[181]:


df_new5=df5[df5[0]!=-1]
df_new5.shape


# In[182]:


Y6=dbscan.labels_
Y_new6=pd.DataFrame(Y6)
Y_new6[0].value_counts()


# In[183]:


df6=pd.concat([df,Y_new6],axis=1)
df6


# In[184]:


df6[df6[0]==-1]


# In[185]:


df_new6=df6[df6[0]!=-1]
df_new6.shape


# In[186]:


Y7=dbscan.labels_
Y_new7=pd.DataFrame(Y7)
Y_new7[0].value_counts()


# In[187]:


df7=pd.concat([df,Y_new7],axis=1)
df7


# In[188]:


df7[df7[0]==-1]


# In[189]:


df_new7=df7[df7[0]!=-1]
df_new7.shape


# In[190]:


Y8=dbscan.labels_
Y_new8=pd.DataFrame(Y8)
Y_new8[0].value_counts()


# In[191]:


df8=pd.concat([df,Y_new8],axis=1)
df8


# In[192]:


df8[df8[0]==-1]


# In[193]:


df_new8=df8[df8[0]!=-1]
df_new8.shape


# In[ ]:





# 2)Perform clustering (hierarchical,K means clustering and DBSCAN) for the airlines data to obtain optimum number of clusters. 
# Draw the inferences from the clusters obtained.
# 
# Data Description:
#  
# The file EastWestAirlinescontains information on passengers who belong to an airline’s frequent flier program. For each passenger the data include information on their mileage history and on different ways they accrued or spent miles in the last year. The goal is to try to identify clusters of passengers that have similar characteristics for the purpose of targeting different segments for different types of mileage offers
# 
# ID --Unique ID
# 
# Balance--Number of miles eligible for award travel
# 
# Qual_mile--Number of miles counted as qualifying for Topflight status
# 
# cc1_miles -- Number of miles earned with freq. flyer credit card in the past 12 months:
# cc2_miles -- Number of miles earned with Rewards credit card in the past 12 months:
# cc3_miles -- Number of miles earned with Small Business credit card in the past 12 months:
# 
# 1 = under 5,000
# 2 = 5,000 - 10,000
# 3 = 10,001 - 25,000
# 4 = 25,001 - 50,000
# 5 = over 50,000
# 
# Bonus_miles--Number of miles earned from non-flight bonus transactions in the past 12 months
# 
# Bonus_trans--Number of non-flight bonus transactions in the past 12 months
# 
# Flight_miles_12mo--Number of flight miles in the past 12 months
# 
# Flight_trans_12--Number of flight transactions in the past 12 months
# 
# Days_since_enrolled--Number of days since enrolled in flier program
# 
# Award--whether that person had award flight (free flight) or not

# In[3]:


df=pd.read_excel('EastWestAirlines.xlsx',sheet_name=1)
df


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


X=df.iloc[:,1:].values
X.shape


# In[7]:


get_ipython().run_line_magic('matplotlib', 'qt')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize']=(16,9)


# In[8]:


from mpl_toolkits.mplot3d import Axes3D

fig1=plt.figure()
ax=Axes3D(fig1)
ax.scatter(X[:,0],X[:,5],X[:,9])
plt.show()


# In[10]:


from mpl_toolkits.mplot3d import Axes3D

fig2=plt.figure()
ax=Axes3D(fig2)
ax.scatter(X[:,2],X[:,3],X[:,4])
plt.show()


# In[11]:


from sklearn.cluster import AgglomerativeClustering


# In[16]:


cluster1=AgglomerativeClustering(n_clusters=6,affinity='euclidean',linkage='ward')
Y1=cluster1.fit_predict(X)
Y_new1=pd.DataFrame(Y1)
Y_new1[0].value_counts()


# In[18]:


plt.figure(figsize=(10,7))
plt.scatter(X[:,0],X[:,5],c=cluster1.labels_,cmap='rainbow')


# In[29]:


cluster2=AgglomerativeClustering(n_clusters=6,affinity='euclidean',linkage='average')
Y2=cluster2.fit_predict(X)
Y_new2=pd.DataFrame(Y2)
Y_new2[0].value_counts()


# In[30]:


plt.figure(figsize=(10,7))
plt.scatter(X[:,0],X[:,9],c=cluster1.labels_,cmap='rainbow')


# In[31]:


cluster3=AgglomerativeClustering(n_clusters=6,affinity='euclidean',linkage='average')
Y3=cluster3.fit_predict(X)
Y_new3=pd.DataFrame(Y3)
Y_new3[0].value_counts()


# In[32]:


plt.figure(figsize=(10,7))
plt.scatter(X[:,5],X[:,9],c=cluster1.labels_,cmap='rainbow')


# In[33]:


Y_clust1=pd.DataFrame(Y1)
Y_clust1[0].value_counts()


# In[34]:


Y_clust2=pd.DataFrame(Y2)
Y_clust2[0].value_counts()


# In[35]:


Y_clust3=pd.DataFrame(Y3)
Y_clust3[0].value_counts()


# In[36]:


from sklearn.cluster import KMeans


# In[38]:


l2=[]
for i in range(1,17):
    kmeans=KMeans(n_clusters=i)
    kmeans=kmeans.fit(X)
    l2.append(kmeans.inertia_)
print(l2)


# In[39]:


import matplotlib.pyplot as plt

plt.scatter(range(1,17),l2)
plt.plot(range(1,17),l2,color='red')
plt.show()


# In[40]:


from sklearn.preprocessing import StandardScaler

SS=StandardScaler()
SS_X=SS.fit_transform(X)
SS_X


# In[41]:


from sklearn.cluster import DBSCAN

dbscan=DBSCAN(eps=1,min_samples=3)
dbscan.fit(SS_X)


# In[43]:


Y1=dbscan.labels_
Y_new1=pd.DataFrame(Y1)
Y_new1[0].value_counts()


# In[44]:


df1=pd.concat([df,Y_new1],axis=1)
df1


# In[45]:


df1[df1[0]==-1]


# In[46]:


df_new1=df1[df1[0]!=-1]
df_new1.shape


# In[47]:


Y2=dbscan.labels_
Y_new2=pd.DataFrame(Y2)
Y_new2[0].value_counts()


# In[48]:


df1=pd.concat([df,Y_new1],axis=1)
df1


# In[49]:


df1[df1[0]==-1]


# In[50]:


df_new1=df1[df1[0]!=-1]
df_new1.shape


# In[51]:


Y3=dbscan.labels_
Y_new3=pd.DataFrame(Y3)
Y_new3[0].value_counts()


# In[53]:


df1=pd.concat([df,Y_new3],axis=1)
df1


# In[54]:


df1[df1[0]==-1]


# In[55]:


df_new1=df1[df1[0]!=-1]
df_new1.shape


# In[ ]:




