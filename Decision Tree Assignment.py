#!/usr/bin/env python
# coding: utf-8

# 1) Decision Tree
#  
# Assignment
# 
# 
# About the data: 
# Let’s consider a Company dataset with around 10 variables and 400 records. 
# The attributes are as follows: 
#  Sales -- Unit sales (in thousands) at each location
#  Competitor Price -- Price charged by competitor at each location
#  Income -- Community income level (in thousands of dollars)
#  Advertising -- Local advertising budget for company at each location (in thousands of dollars)
#  Population -- Population size in region (in thousands)
#  Price -- Price company charges for car seats at each site
#  Shelf Location at stores -- A factor with levels Bad, Good and Medium indicating the quality of the shelving location for the car seats at each site
#  Age -- Average age of the local population
#  Education -- Education level at each location
#  Urban -- A factor with levels No and Yes to indicate whether the store is in an urban or rural location
#  US -- A factor with levels No and Yes to indicate whether the store is in the US or not
# The company dataset looks like this: 
#  
# Problem Statement:
# A cloth manufacturing company is interested to know about the segment or attributes causes high sale. 
# Approach - A decision tree can be built with target variable Sale (we will first convert it in categorical variable) & all other variable will be independent in the analysis.  

# In[2]:


import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[2]:


df=pd.read_csv('Company_Data.csv')
df


# In[3]:


df.describe()


# In[4]:


df['Sales']=pd.cut(x=df['Sales'],bins=[0,3,6,14],labels=['low','medium','high'],right=False)
df['CompPrice']=pd.cut(x=df['CompPrice'],bins=[77,100,135,176],labels=['low','medium','high'],right=False)
df['Income']=pd.cut(x=df['Income'],bins=[21,70,92,121],labels=['low','medium','high'],right=False)
df['Advertising']=pd.cut(x=df['Advertising'],bins=[0,2.5,13,30],labels=['low','medium','high'],right=False)
df['Population']=pd.cut(x=df['Population'],bins=[10,200,400,510],labels=['low','medium','high'],right=False)
df['Price']=pd.cut(x=df['Price'],bins=[24,110,137,195],labels=['low','medium','high'],right=False)
df['Age']=pd.cut(x=df['Age'],bins=[25,48,66,81],labels=['low','medium','high'],right=False)
df['Education']=pd.cut(x=df['Education'],bins=[10,13,16,19],labels=['low','medium','high'],right=False)


# In[5]:


df


# In[6]:


df.shape


# In[7]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(df['ShelveLoc'])
plt.show()
sns.countplot(df['Urban'])
plt.show()
sns.countplot(df['US'])
plt.show()


# In[8]:


sns.countplot(df['Sales'])
plt.show()


# In[9]:


df.info()


# In[10]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()


# In[11]:


for i in df.iloc[:,:]:
    df[i]=LE.fit_transform(df[i])


# In[12]:


df


# In[13]:


Y=df['Sales']
X=df.drop('Sales',axis=1)


# In[14]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier


# In[15]:


a=[]
b=[]
for i in range(1,1000):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
    DTC=DecisionTreeClassifier()
    DTC.fit(X_train,Y_train)
    Y_pred_train=DTC.predict(X_train)
    Y_pred_test=DTC.predict(X_test)
    a.append(accuracy_score(Y_train,Y_pred_train))
    b.append(accuracy_score(Y_test,Y_pred_test))
    
print('Number of Nodes is',DTC.tree_.node_count)
print('Max Depth of a tree is',DTC.tree_.max_depth)
print('Average Accuracy Score for Train Data is',np.mean(a))
print('Average Accuracy Score for Test Data is',np.mean(b))


# In[16]:


a=[]
b=[]
for i in range(1,1000):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
    DTC=DecisionTreeClassifier(criterion='entropy',max_depth=3)
    DTC.fit(X_train,Y_train)
    Y_pred_train=DTC.predict(X_train)
    Y_pred_test=DTC.predict(X_test)
    a.append(accuracy_score(Y_train,Y_pred_train))
    b.append(accuracy_score(Y_test,Y_pred_test))
    
print('Number of Nodes is',DTC.tree_.node_count)
print('Max Depth of a tree is',DTC.tree_.max_depth)
print('Average Accuracy Score for Train Data is',np.mean(a))
print('Average Accuracy Score for Test Data is',np.mean(b))


# In[17]:


import graphviz
from sklearn import tree


# In[18]:


plt.show(tree.plot_tree(DTC))


# In[19]:


fn=['Sales','CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','Us']
cn=['Low','Medium','High']
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(6,6),dpi=600)
plt.show(tree.plot_tree(DTC,feature_names=fn,class_names=cn,filled=True))


# In[20]:


np.mean(Y_pred_test==Y_test)


# In[21]:


a=[]
b=[]
for i in range(1,1000):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
    DTC=DecisionTreeClassifier(criterion='gini',max_depth=3)
    DTC.fit(X_train,Y_train)
    Y_pred_train=DTC.predict(X_train)
    Y_pred_test=DTC.predict(X_test)
    a.append(accuracy_score(Y_train,Y_pred_train))
    b.append(accuracy_score(Y_test,Y_pred_test))
    
print('Number of Nodes is',DTC.tree_.node_count)
print('Max Depth of a tree is',DTC.tree_.max_depth)
print('Average Accuracy Score for Train Data is',np.mean(a))
print('Average Accuracy Score for Test Data is',np.mean(b))


# In[22]:


fn=['Sales','CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','Us']
cn=['Low','Medium','High']
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(6,6),dpi=600)
plt.show(tree.plot_tree(DTC,feature_names=fn,class_names=cn,filled=True))


# In[23]:


np.mean(Y_pred_test==Y_test)


# In[ ]:





# 2) Use decision trees to prepare a model on fraud data 
# treating those who have taxable_income <= 30000 as "Risky" and others are "Good"
# 
# Data Description :
# 
# Undergrad : person is under graduated or not
# Marital.Status : marital status of a person
# Taxable.Income : Taxable income is the amount of how much tax an individual owes to the government 
# Work Experience : Work experience of an individual person
# Urban : Whether that person belongs to urban area or not
# 

# In[36]:


df=pd.read_csv('Fraud_check.csv')
df


# In[37]:


df.describe()


# In[38]:


df['Taxable.Income']=pd.cut(x=df['Taxable.Income'],bins=[0,30000,100000],labels=['Risky','Good'],right=False)
df


# In[39]:


df['Taxable.Income'].value_counts()


# In[40]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(df['Undergrad'])
plt.show()
sns.countplot(df['Marital.Status'])
plt.show()
sns.countplot(df['Urban'])
plt.show()
sns.countplot(df['Taxable.Income'])
plt.show()


# In[41]:


df.info()


# In[42]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

for i in df.iloc[:,:]:
    df[i]=LE.fit_transform(df[i])


# In[43]:


Y=df['Taxable.Income']
X=df.drop('Taxable.Income',axis=1)


# In[44]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier


# In[45]:


c=[]
d=[]
for i in range(1,1000):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
    DTC=DecisionTreeClassifier()
    DTC.fit(X_train,Y_train)
    Y_pred_train=DTC.predict(X_train)
    Y_pred_test=DTC.predict(X_test)
    c.append(accuracy_score(Y_train,Y_pred_train))
    d.append(accuracy_score(Y_test,Y_pred_test))
    
print('Number of Nodes is',DTC.tree_.node_count)
print('Max Depth of a tree is',DTC.tree_.max_depth)
print('Average Accuracy Score for Train Data is',np.mean(c))
print('Average Accuracy Score for Test Data is',np.mean(d))


# In[46]:


c=[]
d=[]
for i in range(1,1000):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
    DTC=DecisionTreeClassifier(criterion='entropy',max_depth=3)
    DTC.fit(X_train,Y_train)
    Y_pred_train=DTC.predict(X_train)
    Y_pred_test=DTC.predict(X_test)
    c.append(accuracy_score(Y_train,Y_pred_train))
    d.append(accuracy_score(Y_test,Y_pred_test))
    
print('Number of Nodes is',DTC.tree_.node_count)
print('Max Depth of a tree is',DTC.tree_.max_depth)
print('Average Accuracy Score for Train Data is',np.mean(c))
print('Average Accuracy Score for Test Data is',np.mean(d))


# In[47]:


import graphviz
from sklearn import tree


# In[48]:


plt.show(tree.plot_tree(DTC))


# In[49]:


fn=['Undergrad','Marital.Status','Taxable.Income','City.Population','Work.Experience','Urban']
cn=['Risky','Good']
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(6,6),dpi=600)
plt.show(tree.plot_tree(DTC,feature_names=fn,class_names=cn,filled=True))


# In[50]:


np.mean(Y_pred_test==Y_test)


# In[51]:


c=[]
d=[]
for i in range(1,1000):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3)
    DTC=DecisionTreeClassifier(criterion='gini',max_depth=3)
    DTC.fit(X_train,Y_train)
    Y_pred_train=DTC.predict(X_train)
    Y_pred_test=DTC.predict(X_test)
    c.append(accuracy_score(Y_train,Y_pred_train))
    d.append(accuracy_score(Y_test,Y_pred_test))
    
print('Number of Nodes is',DTC.tree_.node_count)
print('Max Depth of a tree is',DTC.tree_.max_depth)
print('Average Accuracy Score for Train Data is',np.mean(c))
print('Average Accuracy Score for Test Data is',np.mean(d))


# In[52]:


fn=['Undergrad','Marital.Status','Taxable.Income','City.Population','Work.Experience','Urban']
cn=['Risky','Good']
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(6,6),dpi=600)
plt.show(tree.plot_tree(DTC,feature_names=fn,class_names=cn,filled=True))


# In[53]:


np.mean(Y_pred_test==Y_test)


# In[ ]:




