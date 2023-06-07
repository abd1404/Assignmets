#!/usr/bin/env python
# coding: utf-8

# 1) Prepare a prediction model for profit of 50_startups data.
# Do transformations for getting better predictions of profit and
# make a table containing R^2 value for each prepared model.

# In[3]:


import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.regressionplots import influence_plot
import statsmodels.formula.api as smf
import numpy as np


# In[2]:


startup = pd.read_csv('50_Startups.csv')
startup


# In[3]:


startup.head()


# In[4]:


startup.shape


# In[5]:


startup.info()


# In[6]:


startup.isna().sum()


# In[7]:


startup = startup.rename({'R&D Spend':'RDS', 'Administration':'ADM', 'Marketing Spend':'MS'},axis=1)
startup


# In[8]:


startup.corr()


# In[9]:


sns.set_style(style='darkgrid')
sns.pairplot(startup)


# In[10]:


model = smf.ols('Profit~RDS+ADM+MS', data = startup).fit()


# In[11]:


model.params


# In[12]:


model.tvalues


# In[13]:


model.pvalues


# In[14]:


(model.rsquared,model.rsquared_adj)


# In[15]:


slr_a=smf.ols('Profit~ADM', data = startup).fit()
slr_a.tvalues, slr_a.pvalues
slr_a.summary()


# In[16]:


slr_m = smf.ols('Profit~MS', data = startup).fit()
slr_m.tvalues, slr_m.pvalues
slr_m.summary()


# In[17]:


slr_am = smf.ols('Profit~ADM+MS', data = startup).fit()
slr_am.tvalues, slr_am.pvalues
slr_am.summary()


# In[18]:


rsq_r = smf.ols('RDS~ADM+MS',data = startup).fit().rsquared
vif_r=1/(1-rsq_r)

rsq_a = smf.ols('ADM~RDS+MS', data = startup).fit().rsquared
vif_a = 1/(1-rsq_a)

rsq_m = smf.ols('MS~RDS+ADM', data = startup).fit().rsquared
vif_m = 1/(1-rsq_m)

df1={'Variables':['RDS','ADM','MS'],'Vif':[vif_r,vif_a,vif_m]}
vif_df = pd.DataFrame(df1)
vif_df


# In[19]:


import statsmodels.api as sm
qqplot = sm.qqplot(model.resid, line='q')
plt.title('Normal Q-Q plot of residuals')
plt.show()


# In[20]:


list(np.where(model.resid<-20000))


# In[21]:


def standard_values(vals):
    return (vals-vals.mean())/vals.std()


# In[22]:


plt.scatter(standard_values(model.fittedvalues),standard_values(model.resid))
plt.title('Residual Plot')
plt.xlabel('standardized fitted values')
plt.ylabel('standardized residual values')
plt.show()


# In[23]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, "RDS", fig=fig)
plt.show()


# In[24]:


fig = plt.figure(figsize=(15,8))
fig = sm.graphics.plot_regress_exog(model, 'ADM', fig = fig)
plt.show()


# In[25]:


fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, 'MS', fig = fig)
plt.show()


# In[26]:


model_influence = model.get_influence()
(c,_) = model_influence.cooks_distance


# In[27]:


fig = plt.subplots(figsize=(20,7))
plt.stem(np.arange(len(startup)),np.round(c,3))
plt.xlabel('Row index')
plt.ylabel('Cooks Distance')
plt.show()


# In[28]:


(np.argmax(c), np.max(c))


# In[29]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()


# In[30]:


startup.shape


# In[31]:


k = startup.shape[1]
n = startup.shape[0]
leverage_cutoff = 3*((k + 1 )/n)
leverage_cutoff


# In[32]:


startup[startup.index.isin([49])]


# In[33]:


startup


# In[34]:


startup1 = startup.drop(startup.index[[49]], axis = 0).reset_index(drop=True)
startup1


# In[35]:


final_data = smf.ols('Profit~ RDS+ADM+MS', data = startup1).fit()
final_data.summary()


# In[36]:


(final_data.rsquared, final_data.rsquared_adj)


# In[37]:


new_data = pd.DataFrame({'RDS': 15860,'ADM':58236,'MS':852965}, index= [0])
new_data


# In[38]:


final_data.predict(new_data)


# In[39]:


y_pred = final_data.predict(startup1)
y_pred


# In[40]:


table=pd.DataFrame({'Prep_Models': ['Model','Final_Model'],'Rsquared':[model.rsquared,final_data.rsquared]})
table


# In[ ]:





# 2) Consider only the below columns and prepare a prediction model for predicting Price.
# 
# Corolla<-Corolla[c("Price","Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight")]

# In[4]:


df = pd.read_csv('ToyotaCorolla.csv', encoding = 'ISO-8859-1')
df


# In[5]:


df = pd.concat([df.iloc[:,2:4],df.iloc[:,6],df.iloc[:,8],df.iloc[:,12:14],df.iloc[:,15],df.iloc[:,16:18]], axis = 1)
df


# In[6]:


df.info()


# In[7]:


df.shape


# In[8]:


df[df.duplicated()]


# In[10]:


df = df.drop_duplicates().reset_index(drop=True)
df


# In[11]:


df.describe()


# In[12]:


df.isna().sum()


# In[13]:


df.corr()


# In[14]:


df


# In[15]:


df.isna().sum()


# In[16]:


sns.set_style(style='darkgrid')
sns.pairplot(df)


# In[17]:


model = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight',data = df).fit()


# In[18]:


model.params


# In[19]:


model.tvalues


# In[20]:


model.pvalues


# In[21]:


(model.rsquared,model.rsquared_adj)


# In[22]:


slr_c = smf.ols('Price~cc', data = df).fit()
print(slr_c.tvalues, '\n', slr_c.pvalues)
slr_c.summary()


# In[23]:


slr_d = smf.ols('Price~Doors',data =df).fit()
print(slr_d.tvalues,'\n',slr_d.pvalues)
slr_d.summary()


# In[24]:


slr_cd = smf.ols('Price~cc+Doors', data = df).fit()
print(slr_cd.tvalues,'\n', slr_cd.pvalues)
slr_cd.summary()


# In[25]:


rsq_age = smf.ols('Age_08_04~KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data = df).fit().rsquared
vif_age = 1/(1-rsq_age)

rsq_KM = smf.ols('KM~Age_08_04+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data =df).fit().rsquared
vif_km = 1/(1-rsq_KM)

rsq_HP = smf.ols('HP~Age_08_04+KM+cc+Doors+Gears+Quarterly_Tax+Weight', data = df).fit().rsquared
vif_hp = 1/(1-rsq_HP)

rsq_cc = smf.ols('cc~Age_08_04+KM+HP+Doors+Gears+Quarterly_Tax+Weight', data = df).fit().rsquared
vif_cc = 1/(1-rsq_cc)

rsq_d = smf.ols('Doors~Age_08_04+KM+HP+cc+Gears+Quarterly_Tax+Weight', data = df).fit().rsquared
vif_d = 1/(1-rsq_d)

rsq_g = smf.ols('Gears~Age_08_04+KM+HP+cc+Doors+Quarterly_Tax+Weight', data = df).fit().rsquared
vif_g = 1/(1-rsq_g)

rsq_qt = smf.ols('Quarterly_Tax~Age_08_04+KM+HP+cc+Doors+Gears+Weight', data = df).fit().rsquared
vif_qt = 1/(1-rsq_qt)

rsq_w = smf.ols('Weight~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax', data = df).fit().rsquared
vif_w = 1/(1-rsq_w)

dir1 = pd.DataFrame({'Variable':['Age_08_04','KM','HP','cc','Doors','Gears','Quarterly_Tax','Weight'],
       'VIF':[vif_age, vif_km, vif_hp, vif_cc, vif_d, vif_g, vif_qt, vif_w]})
dir1


# In[26]:


import statsmodels.api as sm
qqplot = sm.qqplot(model.resid, line='q')
plt.title("Normal Q_Q plot of Residuals")
plt.show()


# In[27]:


list(np.where(model.resid>6000))


# In[28]:


list(np.where(model.resid<-6000))


# In[29]:


def standard_values(vals):
    return (vals-vals.mean())/vals.std()


# In[30]:


plt.scatter(standard_values(model.fittedvalues),standard_values(model.resid))
plt.title('Residual plot')
plt.xlabel('Standard Fitted Values')
plt.ylabel('Standard Residual Values')
plt.show()


# In[31]:


fig = plt.figure(figsize = (15,9))
fig = sm.graphics.plot_regress_exog(model,'Age_08_04', fig = fig)
plt.show()


# In[32]:


fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, 'KM', fig = fig)
plt.show()


# In[33]:


fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, 'HP', fig = fig)
plt.show()


# In[34]:


fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, 'cc', fig = fig)
plt.show()


# In[35]:


fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, 'Doors', fig = fig)
plt.show()


# In[36]:


fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, 'Gears', fig = fig)
plt.show()


# In[37]:


fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, 'Quarterly_Tax', fig = fig)
plt.show()


# In[38]:


fig = plt.figure(figsize = (15,8))
fig = sm.graphics.plot_regress_exog(model, 'Weight', fig = fig)
plt.show()


# In[39]:


model_influence = model.get_influence()
(c,_) = model_influence.cooks_distance


# In[40]:


fig = plt.subplots(figsize = (20,7))
plt.stem(np.arange(len(df)), np.round(c,3))
plt.xlabel('Row index')
plt.ylabel("column index")
plt.show()


# In[41]:


(np.argmax(c), np.max(c))


# In[42]:


from statsmodels.graphics.regressionplots import influence_plot
influence_plot(model)
plt.show()


# In[43]:


df.shape


# In[44]:


a = df.shape[1]
s = df.shape[0]
leverage_cutoff = 3*((a+1)/s)


# In[45]:


leverage_cutoff


# In[46]:


df[df.index.isin([80])]


# In[47]:


df1 = df


# In[48]:


df1


# In[50]:


df2 = df1.drop(df1.index[[80]], axis = 0).reset_index(drop=True)
df2


# In[51]:


final_df = smf.ols('Price~Age_08_04+KM+HP+cc+Doors+Gears+Quarterly_Tax+Weight', data = df2).fit()
final_df.summary()


# In[52]:


new_data = pd.DataFrame({'Age_08_04':15,"KM": 58256,'HP':85,"cc": 1500,"Doors": 4,"Gears":7,"Quarterly_Tax":75,'Weight':1500}, index=[0])
new_data


# In[53]:


final_df.predict(new_data)


# In[54]:


y_pred = final_df.predict(df2)
y_pred


# In[55]:


table = pd.DataFrame({'Prep_Models':['Model','Final_Model'],'Rsquared':[model.rsquared,final_df.rsquared]})
table


# In[ ]:




