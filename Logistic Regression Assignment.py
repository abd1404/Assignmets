#!/usr/bin/env python
# coding: utf-8

# Output variable -> y
# y -> Whether the client has subscribed a term deposit or not 
# Binomial ("yes" or "no")

# In[ ]:


import pandas as pd


# In[12]:


df=pd.read_csv("bank-full.csv", sep=';')
df


# In[13]:


df.dtypes


# In[56]:


from sklearn.preprocessing import LabelEncoder,StandardScaler


# In[57]:


LE=LabelEncoder()


# In[58]:


df['y']=LE.fit_transform(df['y'])
df['default']=LE.fit_transform(df['default'])
df['housing']=LE.fit_transform(df['housing'])
df['loan']=LE.fit_transform(df['loan'])
df['job']=LE.fit_transform(df['job'])
df['marital']=LE.fit_transform(df['marital'])
df['education']=LE.fit_transform(df['education'])
df['job']=LE.fit_transform(df['job'])
df['contact']=LE.fit_transform(df['contact'])
df['month']=LE.fit_transform(df['month'])
df['poutcome']=LE.fit_transform(df['poutcome'])


# In[29]:


df


# In[93]:


df.dtypes


# In[94]:


df.shape


# In[32]:


Y=df['y']
X=df.iloc[:,1:16]


# In[33]:


X


# In[76]:


from sklearn.linear_model import LogisticRegression


# In[83]:


Logreg=LogisticRegression(max_iter=30000)
Logreg.fit(X,Y)


# In[84]:


Y_pred=Logreg.predict(X)


# In[85]:


from sklearn.metrics import confusion_matrix,accuracy_score


# In[86]:


CM=confusion_matrix(Y,Y_pred)
CM


# In[89]:


AS=accuracy_score(Y,Y_pred)
print('Accuracy Score is',AS.round(3))


# In[90]:


from sklearn.metrics import recall_score,precision_score,f1_score,roc_curve,roc_auc_score
RS=recall_score(Y,Y_pred)
print('Sensitivity Score is',RS.round(2))

true_negative=CM[0][0]
false_positive=CM[0][1]
def specificity(true_negative,false_positive):
    return(true_negative/(true_negative+false_positive))
print('Specifity Score is',specificity(True_negative,False_positive).round(2))

PS=precision_score(Y,Y_pred)
print('Precision Score is',PS.round(2))

fs=f1_score(Y,Y_pred)
print('F1_score is',fs.round(2))


# In[91]:


pred_prob=Logreg.predict_proba(X)[:,1]
pred_prob


# In[95]:


FPR,TPR,_=roc_curve(Y,pred_prob)
import matplotlib.pyplot as plt
plt.plot(FPR,TPR)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()


# In[55]:


auc=roc_auc_score(Y,pred_prob)
print('Area Under Score:',auc.round(2))


# In[ ]:




