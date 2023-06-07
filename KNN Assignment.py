#!/usr/bin/env python
# coding: utf-8

# 1) Prepare a model for glass classification using KNN
# 
# Data Description:
# 
# RI : refractive index
# 
# Na: Sodium (unit measurement: weight percent in corresponding oxide, as are attributes 4-10)
# 
# Mg: Magnesium
# 
# AI: Aluminum
# 
# Si: Silicon
# 
# K:Potassium
# 
# Ca: Calcium
# 
# Ba: Barium
# 
# Fe: Iron
# 
# Type: Type of glass: (class attribute)
# 1 -- building_windows_float_processed
#  2 --building_windows_non_float_processed
#  3 --vehicle_windows_float_processed
#  4 --vehicle_windows_non_float_processed (none in this database)
#  5 --containers
#  6 --tableware
#  7 --headlamps
# 

# In[2]:


import pandas as pd


# In[3]:


df=pd.read_csv('glass.csv')
df


# In[4]:


Y=df['Type']
X=df.iloc[:,:9]
list(X)


# In[5]:


from sklearn.preprocessing import StandardScaler
SS=StandardScaler()
SS_X=SS.fit_transform(X)
SS_X=pd.DataFrame(SS_X)
SS_X.columns=list(X)
SS_X


# In[6]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(SS_X,Y,train_size=0.7)


# In[10]:


from sklearn.neighbors import KNeighborsClassifier
import warnings


# In[11]:


warnings.filterwarnings('ignore')
KNN=KNeighborsClassifier(n_neighbors=5,p=2)
KNN.fit(X_train,Y_train)

Y_pred_train=KNN.predict(X_train)
Y_pred_test=KNN.predict(X_test)


# In[18]:


from sklearn.metrics import accuracy_score
print('Training Accuracy Score is',accuracy_score(Y_train,Y_pred_train).round(2))
print('Test Accuracy Score is',accuracy_score(Y_test,Y_pred_test).round(2))


# In[14]:


warnings.filterwarnings('ignore')
KNN=KNeighborsClassifier(n_neighbors=7,p=1)
KNN.fit(X_train,Y_train)

Y_pred_train=KNN.predict(X_train)
Y_pred_test=KNN.predict(X_test)


# In[15]:


from sklearn.metrics import accuracy_score
print('Training Accuracy Score is',accuracy_score(Y_train,Y_pred_train).round(2))
print('Test Accuracy Score is',accuracy_score(Y_test,Y_pred_test).round(2))


# In[16]:


warnings.filterwarnings('ignore')
KNN=KNeighborsClassifier(n_neighbors=9,p=2)
KNN.fit(X_train,Y_train)

Y_pred_train=KNN.predict(X_train)
Y_pred_test=KNN.predict(X_test)


# In[17]:


from sklearn.metrics import accuracy_score
print('Training Accuracy Score is',accuracy_score(Y_train,Y_pred_train).round(2))
print('Test Accuracy Score is',accuracy_score(Y_test,Y_pred_test).round(2))


# In[19]:


warnings.filterwarnings('ignore')
KNN=KNeighborsClassifier(n_neighbors=11,p=1)
KNN.fit(X_train,Y_train)

Y_pred_train=KNN.predict(X_train)
Y_pred_test=KNN.predict(X_test)


# In[20]:


from sklearn.metrics import accuracy_score
print('Training Accuracy Score is',accuracy_score(Y_train,Y_pred_train).round(2))
print('Test Accuracy Score is',accuracy_score(Y_test,Y_pred_test).round(2))


# In[ ]:





# 2) Implement a KNN model to classify the animals in to categorie

# In[12]:


import pandas as pd


# In[24]:


df=pd.read_csv('Zoo.csv')
df


# In[25]:


from sklearn.preprocessing import LabelEncoder,StandardScaler
LE=LabelEncoder()
df['animal name']=LE.fit_transform(df['animal name'])
df['animal name']


# In[26]:


Y=df['type']
X=df.iloc[:,0:17]
list(X)


# In[27]:


SS=StandardScaler()
SS_X=SS.fit_transform(X)
SS_X=pd.DataFrame(SS_X)
SS_X.columns=list(X)
SS_X


# In[28]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(SS_X,Y,train_size=0.7)


# In[29]:


from sklearn.neighbors import KNeighborsClassifier
import warnings


# In[30]:


warnings.filterwarnings('ignore')
KNN=KNeighborsClassifier(n_neighbors=5,p=2)
KNN.fit(X_train,Y_train)

Y_pred_train=KNN.predict(X_train)
Y_pred_test=KNN.predict(X_test)


# In[31]:


from sklearn.metrics import accuracy_score
print('Training Accuracy Score is',accuracy_score(Y_train,Y_pred_train).round(2))
print('Test Accuracy Score is',accuracy_score(Y_test,Y_pred_test).round(2))


# In[34]:


warnings.filterwarnings('ignore')
KNN=KNeighborsClassifier(n_neighbors=9,p=1)
KNN.fit(X_train,Y_train)

Y_pred_train=KNN.predict(X_train)
Y_pred_test=KNN.predict(X_test)


# In[35]:


from sklearn.metrics import accuracy_score
print('Training Accuracy Score is',accuracy_score(Y_train,Y_pred_train).round(2))
print('Test Accuracy Score is',accuracy_score(Y_test,Y_pred_test).round(2))


# In[36]:


warnings.filterwarnings('ignore')
KNN=KNeighborsClassifier(n_neighbors=13,p=2)
KNN.fit(X_train,Y_train)

Y_pred_train=KNN.predict(X_train)
Y_pred_test=KNN.predict(X_test)


# In[37]:


from sklearn.metrics import accuracy_score
print('Training Accuracy Score is',accuracy_score(Y_train,Y_pred_train).round(2))
print('Test Accuracy Score is',accuracy_score(Y_test,Y_pred_test).round(2))


# In[38]:


warnings.filterwarnings('ignore')
KNN=KNeighborsClassifier(n_neighbors=15,p=2)
KNN.fit(X_train,Y_train)

Y_pred_train=KNN.predict(X_train)
Y_pred_test=KNN.predict(X_test)


# In[39]:


from sklearn.metrics import accuracy_score
print('Training Accuracy Score is',accuracy_score(Y_train,Y_pred_train).round(2))
print('Test Accuracy Score is',accuracy_score(Y_test,Y_pred_test).round(2))


# In[ ]:




