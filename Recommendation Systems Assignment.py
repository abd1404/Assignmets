#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[3]:


df=pd.read_csv('book.csv',encoding ='iso-8859-1')
df


# In[4]:


df.shape


# In[5]:


df.head()


# In[6]:


df.sort_values('User.ID')
len(df)


# In[7]:


len(df['User.ID'].unique())


# In[8]:


len(df['Book.Title'].unique())


# In[9]:


df['Book.Rating'].value_counts()


# In[10]:


df['Book.Rating'].hist()


# In[11]:


user_df = df.pivot_table(index='User.ID',columns='Book.Title',values='Book.Rating')
user_df


# In[15]:


user_df.fillna(0, inplace=True)
user_df


# In[26]:


from sklearn.metrics import pairwise_distances
user_sim = 1 - pairwise_distances(user_df,metric='cosine')


# In[27]:


user_sim.shape


# In[28]:


user_sim_df = pd.DataFrame(user_sim)
user_sim_df


# In[31]:


user_sim_df.index   = df['User.ID'].unique()
user_sim_df.columns = df['User.ID'].unique()


# In[33]:


user_sim_df.head()


# In[34]:


user_sim_df.shape


# In[35]:


np.fill_diagonal(user_sim, 0)


# In[38]:


user_sim_df.idxmax(axis=1)[0:10]


# In[41]:


df[(df['User.ID']==276729) | (df['User.ID']==276726)]


# In[42]:


df[(df['User.ID']==276744) | (df['User.ID']==276726)]


# In[ ]:




