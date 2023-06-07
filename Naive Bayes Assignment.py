#!/usr/bin/env python
# coding: utf-8

# 1) Prepare a classification model using Naive Bayes 
# for salary data 
# 
# Data Description:
# 
# age -- age of a person
# workclass	-- A work class is a grouping of work 
# education	-- Education of an individuals	
# maritalstatus -- Marital status of an individulas	
# occupation	 -- occupation of an individuals
# relationship -- 	
# race --  Race of an Individual
# sex --  Gender of an Individual
# capitalgain --  profit received from the sale of an investment	
# capitalloss	-- A decrease in the value of a capital asset
# hoursperweek -- number of hours work per week	
# native -- Native of an individual
# Salary -- salary of an individual

# In[1]:


import pandas as pd


# In[2]:


df=pd.read_csv('SalaryData_Train.csv')
df


# In[3]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()


# In[4]:


for i in range(0,14):
    df.iloc[:,i]=LE.fit_transform(df.iloc[:,i])


# In[7]:


Y=df['Salary']
X=df.iloc[:,:13]


# In[9]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)


# In[10]:


from sklearn.naive_bayes import MultinomialNB
MNB=MultinomialNB()
MNB.fit(X_train,Y_train)
Y_pred_train=MNB.predict(X_train)
Y_pred_test=MNB.predict(X_test)


# In[12]:


from sklearn.metrics import accuracy_score
print('Training Accuracy is',accuracy_score(Y_train,Y_pred_train).round(2))
print('Test Accuracy is',accuracy_score(Y_test,Y_pred_test).round(2))


# In[ ]:





# 2)

# In[13]:


df=pd.read_csv('SalaryData_Test.csv')
df


# In[14]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()


# In[19]:


for i in range(0,14):
    df.iloc[:,i]=LE.fit_transform(df.iloc[:,i])


# In[20]:


Y=df['Salary']
X=df.iloc[:,:13]


# In[21]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)


# In[22]:


from sklearn.naive_bayes import MultinomialNB
MNB=MultinomialNB()
MNB.fit(X_train,Y_train)
Y_pred_train=MNB.predict(X_train)
Y_pred_test=MNB.predict(X_test)


# In[23]:


from sklearn.metrics import accuracy_score
print('Training Accuracy is',accuracy_score(Y_train,Y_pred_train).round(2))
print('Test Accuracy is',accuracy_score(Y_test,Y_pred_test).round(2))


# In[ ]:




