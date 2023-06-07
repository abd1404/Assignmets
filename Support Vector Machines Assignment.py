#!/usr/bin/env python
# coding: utf-8

# 1) Prepare a classification model using SVM for salary data 
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
import numpy as np
import warnings
warnings.filterwarnings('ignore')


# In[2]:


train_data=pd.read_csv('SalaryData_Train(1).csv')
test_data=pd.read_csv('SalaryData_Test(1).csv')


# In[3]:


train_data


# In[4]:


test_data


# In[5]:


train_data.describe()


# In[6]:


train_data.corr()


# In[7]:


test_data.describe()


# In[8]:


test_data.corr()


# In[9]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(train_data['workclass'])
plt.show()
sns.countplot(train_data['relationship'])
plt.show()
sns.countplot(train_data['race'])
plt.show()
sns.countplot(train_data['sex'])
plt.show()
sns.countplot(train_data['Salary'])
plt.show()


# In[10]:


sns.countplot(test_data['workclass'])
plt.show()
sns.countplot(test_data['relationship'])
plt.show()
sns.countplot(test_data['race'])
plt.show()
sns.countplot(test_data['sex'])
plt.show()
sns.countplot(test_data['Salary'])
plt.show()


# In[11]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()


# In[12]:


train_data['workclass']=LE.fit_transform(train_data['workclass'])
train_data['education']=LE.fit_transform(train_data['education'])
train_data['maritalstatus']=LE.fit_transform(train_data['maritalstatus'])
train_data['occupation']=LE.fit_transform(train_data['occupation'])
train_data['relationship']=LE.fit_transform(train_data['relationship'])
train_data['race']=LE.fit_transform(train_data['sex'])
train_data['native']=LE.fit_transform(train_data['native'])
train_data['sex']=LE.fit_transform(train_data['sex'])
train_data['Salary']=LE.fit_transform(train_data['Salary'])


# In[13]:


train_data


# In[14]:


test_data['workclass']=LE.fit_transform(test_data['workclass'])
test_data['education']=LE.fit_transform(test_data['education'])
test_data['maritalstatus']=LE.fit_transform(test_data['maritalstatus'])
test_data['occupation']=LE.fit_transform(test_data['occupation'])
test_data['relationship']=LE.fit_transform(test_data['relationship'])
test_data['race']=LE.fit_transform(test_data['sex'])
test_data['native']=LE.fit_transform(test_data['native'])
test_data['sex']=LE.fit_transform(test_data['sex'])
test_data['Salary']=LE.fit_transform(test_data['Salary'])


# In[15]:


test_data


# In[16]:


Y_train=train_data['Salary']
X_train=train_data.drop('Salary',axis=1)
Y_test=test_data['Salary']
X_test=test_data.drop('Salary',axis=1)


# In[17]:


from sklearn.svm import SVC

clf=SVC()
clf.fit(X_train,Y_train)
Y_pred_train=clf.predict(X_train)
Y_pred_test=clf.predict(X_test)


# In[18]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[20]:


cm=confusion_matrix(Y_train,Y_pred_train)
cm1=confusion_matrix(Y_test,Y_pred_test)

print('Training Confusion Matrix is',cm)
print('Test Confusion Matrix is',cm1)


# In[21]:


print('Training Accuracy Score is',accuracy_score(Y_train,Y_pred_train).round(3))
print('Test Accuracy Score is',accuracy_score(Y_test,Y_pred_test).round(3))


# In[ ]:


params = [{'kernel':['linear'],'gamma':[50,10,5,1,0.5,0.1],'C':[15,14,13,12,11,10,5,1,0.1,0.01,0.001]}]
from sklearn.model_selection import GridSearchCV
GSC = GridSearchCV(clf,params,cv=10)
GSC.fit(X_train,Y_train)
GSC.best_score_ 


# In[ ]:


GSC.best_params_


# In[ ]:


svc = SVC(c=,gamma=)
svc.fit(X_train,Y_train)
Y_pred_train = svc.predict(X_train)
Y_pred_test = svc.predict(X_test)
print("Accuracy Score for train data : ",accuracy_score(Y_train,Y_pred_train))
print("Accuracy Score for test data : ", accuracy_score(Y_test,Y_pred_test))


# In[ ]:





# 2) classify the Size_Categorie using SVM
# 
# month	month of the year: 'jan' to 'dec'
# day	day of the week: 'mon' to 'sun'
# FFMC	FFMC index from the FWI system: 18.7 to 96.20
# DMC	DMC index from the FWI system: 1.1 to 291.3
# DC	DC index from the FWI system: 7.9 to 860.6
# ISI	ISI index from the FWI system: 0.0 to 56.10
# temp	temperature in Celsius degrees: 2.2 to 33.30
# RH	relative humidity in %: 15.0 to 100
# wind	wind speed in km/h: 0.40 to 9.40
# rain	outside rain in mm/m2 : 0.0 to 6.4
# Size_Categorie 	the burned area of the forest ( Small , Large)

# In[2]:


df=pd.read_csv('forestfires.csv')
df


# In[3]:


df.describe()


# In[4]:


df.corr()


# In[6]:


import matplotlib.pyplot as plt
import seaborn as sns
sns.countplot(df['month'])
plt.show()
sns.countplot(df['day'])
plt.show()
sns.countplot(df['size_category'])
plt.show()


# In[7]:


from sklearn.preprocessing import LabelEncoder
LE=LabelEncoder()


# In[9]:


df['month']=LE.fit_transform(df['month'])
df['day']=LE.fit_transform(df['day'])
df['size_category']=LE.fit_transform(df['size_category'])
df


# In[12]:


Y=df['size_category']
X=df.drop('size_category',axis=1)


# In[19]:


from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y)


# In[20]:


from sklearn.svm import SVC

clf=SVC()
clf.fit(X,Y)
Y_pred_train=clf.predict(X_train)
Y_pred_test=clf.predict(X_test)


# In[21]:


from sklearn.metrics import accuracy_score,confusion_matrix


# In[23]:


cm=confusion_matrix(Y_train,Y_pred_train)
cm1=confusion_matrix(Y_test,Y_pred_test)

print('Training Confusion Matrix is',cm)
print('Test Confusion Matrix is',cm1)


# In[24]:


print('Training Accuracy Score is',accuracy_score(Y_train,Y_pred_train).round(3))
print('Test Accuracy Score is',accuracy_score(Y_test,Y_pred_test).round(3))


# In[25]:


params = [{'kernel':['linear'],'gamma':[50,10,5,1,0.5,0.1],'C':[15,14,13,12,11,10,5,1,0.1,0.01,0.001]}]
from sklearn.model_selection import GridSearchCV
GSC = GridSearchCV(clf,params,cv=10)
GSC.fit(X_train,Y_train)
GSC.best_score_ 


# In[26]:


GSC.best_params_


# In[29]:


svc = SVC(kernel='linear',C=15,gamma=50)
svc.fit(X_train,Y_train)
Y_pred_train = svc.predict(X_train)
Y_pred_test = svc.predict(X_test)
print("Accuracy Score for train data : ",accuracy_score(Y_train,Y_pred_train))
print("Accuracy Score for test data : ", accuracy_score(Y_test,Y_pred_test))


# In[ ]:




