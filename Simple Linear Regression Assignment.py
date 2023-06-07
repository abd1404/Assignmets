#!/usr/bin/env python
# coding: utf-8

# 1) Delivery_time -> Predict delivery time using sorting time

# In[18]:


import numpy as np
import pandas as pd


# In[19]:


df=pd.read_csv('delivery_time.csv')
df


# In[20]:


X=df[['Sorting Time']]
Y=df['Delivery Time']

import matplotlib.pyplot as plt
plt.scatter(X,Y)
plt.xlabel('Sorting Time')
plt.ylabel('Delivery Time')
plt.show()


# In[21]:


from sklearn.linear_model import LinearRegression
LR=LinearRegression()
LR.fit(X,Y)


# In[22]:


LR.intercept_


# In[23]:


LR.coef_


# In[24]:


Y_pred=LR.predict(X)


# In[25]:


import matplotlib.pyplot as plt
plt.scatter(X,Y)
plt.scatter(X,Y_pred,color='red')
plt.plot(X,Y_pred,color='black')
plt.show()


# In[9]:


from sklearn.metrics import mean_squared_error


# In[10]:


mse=mean_squared_error(Y,Y_pred)
print('Mean Squared Error:',mse.round(2))


# In[11]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error:',rmse.round(2))


# In[31]:


X1=pow(X,2)
LR.fit(X1,Y)
Y_pred=LR.predict(X1)
plt.scatter(X1,Y,color='blue')
plt.scatter(X1,Y_pred,color='red')
plt.plot(X1,Y_pred, color='black')
plt.show()

mse=mean_squared_error(Y,Y_pred)
rmse=np.sqrt(mse)
print("Mean Squared Error: ", mse.round(2))
print("Root Mean Square Error: ", rmse.round(2))


# In[26]:


X2=np.sqrt(X)
LR.fit(X2,Y)
Y_pred=LR.predict(X2)
plt.scatter(X2,Y,color='blue')
plt.scatter(X2,Y_pred,color='red')
plt.plot(X2,Y_pred, color='black')
plt.show()

mse=mean_squared_error(Y,Y_pred)
rmse=np.sqrt(mse)
print("Mean Squared Error: ", mse)
print("Root Mean Square Error: ", rmse)


# In[28]:


X3=np.log(X)
LR.fit(X3,Y)
Y_pred=LR.predict(X3)
plt.scatter(X3,Y,color='blue')
plt.scatter(X3,Y_pred,color='red')
plt.plot(X3,Y_pred, color='black')
plt.show()

mse=mean_squared_error(Y,Y_pred)
rmse=np.sqrt(mse)
print("Mean Squared Error: ", mse)
print("Root Mean Square Error: ", rmse)


# In[30]:


X4=(1/X)
LR.fit(X4,Y)
Y_pred=LR.predict(X4)
plt.scatter(X4,Y,color='blue')
plt.scatter(X4,Y_pred,color='red')
plt.plot(X4,Y_pred, color='black')
plt.show()

mse=mean_squared_error(Y,Y_pred)
rmse=np.sqrt(mse)
print("Mean Squared Error: ", mse)
print("Root Mean Square Error: ", rmse)


# In[32]:


X5=pow(X,3)
LR.fit(X5,Y)
Y_pred=LR.predict(X5)
plt.scatter(X5,Y,color='blue')
plt.scatter(X5,Y_pred,color='red')
plt.plot(X5,Y_pred, color='black')
plt.show()

mse=mean_squared_error(Y,Y_pred)
rmse=np.sqrt(mse)
print("Mean Squared Error: ", mse)
print("Root Mean Square Error: ", rmse)


# In[33]:


X6=np.sqrt(1/X)
LR.fit(X6,Y)
Y_pred=LR.predict(X6)
plt.scatter(X6,Y,color='blue')
plt.scatter(X6,Y_pred,color='red')
plt.plot(X6,Y_pred, color='black')
plt.show()

mse=mean_squared_error(Y,Y_pred)
rmse=np.sqrt(mse)
print("Mean Squared Error: ", mse)
print("Root Mean Square Error: ", rmse)


# Hence, Simple Regression Model for prediction of delivery time using sorting time is prepared.

# In[ ]:





# 2) Salary_hike -> Build a prediction model for Salary_hike
# 

# In[34]:


df=pd.read_csv('Salary_Data.csv')
df


# In[35]:


Y=df['Salary']
X=df[['YearsExperience']]


# In[36]:


import matplotlib.pyplot as plt
plt.scatter(X,Y)
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()


# In[37]:


from sklearn.linear_model import LinearRegression


# In[38]:


LR=LinearRegression()
LR.fit(X,Y)


# In[39]:


LR.intercept_


# In[40]:


LR.coef_


# In[41]:


Y_pred=LR.predict(X)


# In[42]:


plt.scatter(X,Y)
plt.scatter(X,Y_pred,color='red')
plt.plot(X,Y_pred,color='black')
plt.xlabel('YearsExperience')
plt.ylabel('Salary')
plt.show()


# In[43]:


from sklearn.metrics import mean_squared_error


# In[44]:


mse=mean_squared_error(Y,Y_pred)
print('Mean Squared Error:',mse.round(2))


# In[45]:


rmse=np.sqrt(mse)
print('Root Mean Squared Error:',rmse.round(2))


# In[46]:


X1=pow(X,2)
LR.fit(X1,Y)
Y_pred=LR.predict(X1)
plt.scatter(X1,Y,color='blue')
plt.scatter(X1,Y_pred,color='red')
plt.plot(X1,Y_pred, color='black')
plt.show()

mse=mean_squared_error(Y,Y_pred)
rmse=np.sqrt(mse)
print("Mean Squared Error: ", mse.round(2))
print("Root Mean Square Error: ", rmse.round(2))


# In[47]:


X2=np.sqrt(X)
LR.fit(X2,Y)
Y_pred=LR.predict(X2)
plt.scatter(X2,Y,color='blue')
plt.scatter(X2,Y_pred,color='red')
plt.plot(X2,Y_pred, color='black')
plt.show()

mse=mean_squared_error(Y,Y_pred)
rmse=np.sqrt(mse)
print("Mean Squared Error: ", mse)
print("Root Mean Square Error: ", rmse)


# In[48]:


X3=np.log(X)
LR.fit(X3,Y)
Y_pred=LR.predict(X3)
plt.scatter(X3,Y,color='blue')
plt.scatter(X3,Y_pred,color='red')
plt.plot(X3,Y_pred, color='black')
plt.show()

mse=mean_squared_error(Y,Y_pred)
rmse=np.sqrt(mse)
print("Mean Squared Error: ", mse)
print("Root Mean Square Error: ", rmse)


# In[49]:


X4=(1/X)
LR.fit(X4,Y)
Y_pred=LR.predict(X4)
plt.scatter(X4,Y,color='blue')
plt.scatter(X4,Y_pred,color='red')
plt.plot(X4,Y_pred, color='black')
plt.show()

mse=mean_squared_error(Y,Y_pred)
rmse=np.sqrt(mse)
print("Mean Squared Error: ", mse)
print("Root Mean Square Error: ", rmse)


# In[50]:


X5=pow(X,3)
LR.fit(X5,Y)
Y_pred=LR.predict(X5)
plt.scatter(X5,Y,color='blue')
plt.scatter(X5,Y_pred,color='red')
plt.plot(X5,Y_pred, color='black')
plt.show()

mse=mean_squared_error(Y,Y_pred)
rmse=np.sqrt(mse)
print("Mean Squared Error: ", mse)
print("Root Mean Square Error: ", rmse)


# In[51]:


X6=np.sqrt(1/X)
LR.fit(X6,Y)
Y_pred=LR.predict(X6)
plt.scatter(X6,Y,color='blue')
plt.scatter(X6,Y_pred,color='red')
plt.plot(X6,Y_pred, color='black')
plt.show()

mse=mean_squared_error(Y,Y_pred)
rmse=np.sqrt(mse)
print("Mean Squared Error: ", mse)
print("Root Mean Square Error: ", rmse)


# Hence, Simple Regression Model for prediction of Salary using YearsExperience is prepared.

# In[ ]:




