#!/usr/bin/env python
# coding: utf-8

# 1) Random Forest
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
# Approach - A Random Forest can be built with target variable Sales (we will first convert it in categorical variable) & all other variable will be independent in the analysis.  

# In[1]:


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


# In[8]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(df['ShelveLoc'])
plt.show()
sns.countplot(df['Urban'])
plt.show()
sns.countplot(df['US'])
plt.show()
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


# In[15]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[16]:


a=[]
b=[]
for i in range(1,1000):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4)
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


# In[17]:


a=[]
b=[]
for i in range(1,1000):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4)
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


# In[18]:


import graphviz
from sklearn import tree


# In[19]:


plt.show(tree.plot_tree(DTC))


# In[20]:


fn=['Sales','CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','Us']
cn=['Low','Medium','High']
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(6,6),dpi=600)
plt.show(tree.plot_tree(DTC,feature_names=fn,class_names=cn,filled=True))


# In[21]:


np.mean(Y_pred_test==Y_test)


# In[22]:


a=[]
b=[]
for i in range(1,1000):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4)
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


# In[23]:


fn=['Sales','CompPrice','Income','Advertising','Population','Price','ShelveLoc','Age','Education','Urban','Us']
cn=['Low','Medium','High']
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(6,6),dpi=600)
plt.show(tree.plot_tree(DTC,feature_names=fn,class_names=cn,filled=True))


# In[24]:


np.mean(Y_pred_test==Y_test)


# In[25]:


RFC=RandomForestClassifier(n_estimators=500,max_samples=0.7,max_features=0.7,random_state=27)


# In[51]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=27)

RFC.fit(X_train,Y_train)

Y_pred_train=RFC.predict(X_train)
Y_pred_test=RFC.predict(X_test)


# In[52]:


print('Random Forest Training Accuracy Score is',accuracy_score(Y_train,Y_pred_train).round(2))
print('Random Forest Test Accuracy Score is',accuracy_score(Y_test,Y_pred_test).round(2))


# In[ ]:





# 2) Use Random Forest to prepare a model on fraud data 
# treating those who have taxable_income <= 30000 as "Risky" and others are "Good"

# In[29]:


df=pd.read_csv('Fraud_check.csv')
df


# In[30]:


df.describe()


# In[31]:


df['Taxable.Income']=pd.cut(x=df['Taxable.Income'],bins=[0,30000,100000],labels=['Risky','Good'],right=False)
df


# In[32]:


df['Taxable.Income'].value_counts()


# In[33]:


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


# In[34]:


df.info()


# In[35]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()

for i in df.iloc[:,:]:
    df[i]=LE.fit_transform(df[i])


# In[36]:


Y=df['Taxable.Income']
X=df.drop('Taxable.Income',axis=1)


# In[37]:


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier


# In[38]:


c=[]
d=[]
for i in range(1,1000):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4)
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


# In[39]:


c=[]
d=[]
for i in range(1,1000):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4)
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


# In[40]:


import graphviz
from sklearn import tree


# In[41]:


plt.show(tree.plot_tree(DTC))


# In[42]:


fn=['Undergrad','Marital.Status','Taxable.Income','City.Population','Work.Experience','Urban']
cn=['Risky','Good']
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(6,6),dpi=600)
plt.show(tree.plot_tree(DTC,feature_names=fn,class_names=cn,filled=True))


# In[43]:


np.mean(Y_pred_test==Y_test)


# In[46]:


c=[]
d=[]
for i in range(1,1000):
    X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.4)
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


# In[47]:


fn=['Undergrad','Marital.Status','Taxable.Income','City.Population','Work.Experience','Urban']
cn=['Risky','Good']
fig,axes=plt.subplots(nrows=1,ncols=1,figsize=(6,6),dpi=600)
plt.show(tree.plot_tree(DTC,feature_names=fn,class_names=cn,filled=True))


# In[48]:


np.mean(Y_pred_test==Y_test)


# In[49]:


RFC=RandomForestClassifier(n_estimators=500,max_samples=0.7,max_features=0.7,random_state=29)


# In[56]:


X_train,X_test,Y_train,Y_test=train_test_split(X,Y,random_state=29)

RFC.fit(X_train,Y_train)

Y_pred_train=RFC.predict(X_train)
Y_pred_test=RFC.predict(X_test)


# In[57]:


print('Random Forest Training Accuracy Score is',accuracy_score(Y_train,Y_pred_train).round(2))
print('Random Forest Test Accuracy Score is',accuracy_score(Y_test,Y_pred_test).round(2))


# In[ ]:




