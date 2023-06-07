#!/usr/bin/env python
# coding: utf-8

# Forecast the CocaCola prices and Airlines Passengers data set. Prepare a document for each model explaining 
# how many dummy variables you have created and RMSE value for each model. Finally which model you will use for 
# Forecasting.

# 1) CocaCola Prices

# In[143]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')


from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.holtwinters import SimpleExpSmoothing # SES
from statsmodels.tsa.holtwinters import Holt # Holts Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from sklearn.metrics import mean_squared_error

import itertools
import statsmodels.api as sm


# In[144]:


coke=pd.read_excel("CocaCola_Sales_Rawdata.xlsx",index_col=0,parse_dates=True) 
coke


# In[145]:


coke.info()


# In[146]:


coke = pd.read_excel("CocaCola_Sales_Rawdata.xlsx", index_col = 0,header = 0,parse_dates = True)
coke


# In[147]:


coke.index


# In[148]:


plt.figure(figsize = (15,7))
plt.plot(coke)


# In[149]:


coke.plot(kind='kde')


# In[150]:


coke.hist()


# In[151]:


plt.figure(figsize = (17,7))
pd.plotting.lag_plot(coke)


# In[152]:


from statsmodels.graphics.tsaplots import plot_acf

plt.figure(figsize = (50,15))
plot_acf(coke, lags=6)
plt.show()


# In[153]:


coke = pd.read_excel("CocaCola_Sales_Rawdata.xlsx",index_col = 0,header = 0,parse_dates = True, squeeze=True)
coke


# In[154]:


type(coke)


# In[155]:


coke = pd.read_excel("CocaCola_Sales_Rawdata.xlsx",index_col = 0,header = 0,parse_dates = True,squeeze=True)
coke


# In[156]:


coke.shape


# In[157]:


coke = pd.read_excel("CocaCola_Sales_Rawdata.xlsx")


# In[158]:


quarter =['Q1','Q2','Q3','Q4']


# In[159]:


p = coke["Quarter"][0]
p[0:2]
coke['quarter']= 0

for i in range(42):
    p = coke["Quarter"][i]
    coke['quarter'][i]= p[0:2]

coke


# In[160]:


quarter_dummies = pd.DataFrame(pd.get_dummies(coke['quarter']))
quarter_dummies


# In[161]:


coke=pd.concat([coke,quarter_dummies],axis=1)
coke


# In[162]:


coke['t']=np.arange(1,43)
coke['t_square']=np.square(coke.t)
coke['log_Sales']=np.log(coke.Sales)
coke


# In[163]:


coke


# In[164]:


coke['Sales'].plot()


# In[165]:


plt.figure(figsize=(12,4))
sns.lineplot(x="quarter",y="Sales",data=coke)


# In[166]:


coke


# In[167]:


Train = coke.head(25)
Test = coke.tail(7)


# In[168]:


Train


# In[169]:


Test


# In[170]:


#Linear Model
import statsmodels.formula.api as smf 

linear_model = smf.ols('Sales~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_linear))**2))
rmse_linear


# In[171]:


#Exponential
Exp = smf.ols('log_Sales~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


# In[172]:


#Quadratic 
Quad = smf.ols('Sales~t+t_square',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_Quad))**2))
rmse_Quad


# In[173]:


#Additive seasonality 
add_sea = smf.ols('Sales~Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Q1', 'Q2', 'Q3', 'Q4']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea))**2))
rmse_add_sea


# In[174]:


#Additive Seasonality Quadratic 
add_sea_Quad = smf.ols('Sales~t+t_square+Q1+Q2+Q3+Q4',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Q1', 'Q2', 'Q3', 'Q4','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad


# In[175]:


##Multiplicative Seasonality
Mul_sea = smf.ols('log_Sales~Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


# In[176]:


#Multiplicative Additive Seasonality 
Mul_Add_sea = smf.ols('log_Sales~t+Q1+Q2+Q3+Q4',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea


# In[177]:


#Multiplicative Seasonality Quadratic 
mult_sea_Quad = smf.ols('log_Sales~t+t_square+Q1+Q2+Q3+Q4',data=Train).fit()
pred_mult_sea_quad = pd.Series(mult_sea_Quad.predict(Test[['Q1', 'Q2', 'Q3', 'Q4','t','t_square']]))
rmse_mult_sea_quad = np.sqrt(np.mean((np.array(Test['Sales'])-np.array(pred_mult_sea_quad))**2))
rmse_mult_sea_quad


# In[178]:


data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])


# In[179]:


model_full = smf.ols('Sales~t+t_square+Q1+Q2+Q3+Q4',data=coke).fit()


# In[180]:


pred_new  = pd.Series(model_full.predict(coke))
pred_new


# In[181]:


coke["forecasted_Sales"] = pd.Series(np.exp(pred_new))


# In[182]:


plt.figure(figsize=(18,10))
plt.plot(coke[['Sales','forecasted_Sales']].reset_index(drop=True))


# In[183]:


Train = coke.head(35)
Test = coke.tail(7)


# In[184]:


plt.figure(figsize=(24,7))
coke['Sales'].plot(label="org")
coke["Sales"].rolling(4).mean().plot(label=str(5))
plt.legend(loc='best')


# In[185]:


plt.figure(figsize=(24,7))
coke['Sales'].plot(label="org")
for i in range(2,18,6):
    coke["Sales"].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')


# In[186]:


decompose_ts_add = seasonal_decompose(coke['Sales'], period = 12)
decompose_ts_add.plot()
plt.show()


# In[187]:


import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(coke.Sales,lags=12)
tsa_plots.plot_pacf(coke.Sales,lags=12)
plt.show()


# In[188]:


def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


# In[189]:


ses_model = SimpleExpSmoothing(Train["Sales"]).fit(smoothing_level=0.2)
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Sales) 


# In[190]:


hw_model = Holt(Train["Sales"]).fit(smoothing_level=0.8, smoothing_slope=0.2)
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Sales) 


# In[191]:


hwe_model_add_add = ExponentialSmoothing(Train["Sales"],seasonal="add",trend="add",seasonal_periods=12).fit() #add the trend to the model
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Sales)


# In[192]:


hwe_model_mul_add = ExponentialSmoothing(Train["Sales"],seasonal="mul",trend="add",seasonal_periods=12).fit() 
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Sales)


# In[193]:


hwe_model_mul_add = ExponentialSmoothing(coke["Sales"],seasonal="mul",trend="add",seasonal_periods=12).fit() 


# In[194]:


hwe_model_mul_add.forecast(7)


# In[ ]:





# 2) Airline Passengers Data Set

# In[195]:


airlines = pd.read_excel('Airlines+Data.xlsx',index_col=0,parse_dates=['Month'])
airlines


# In[196]:


airlines.info()


# In[197]:


airlines.index


# In[198]:


plt.figure(figsize = (15,7))
plt.plot(airlines)


# In[199]:


airlines = pd.read_excel("Airlines+Data.xlsx",index_col = 0,header = 0, parse_dates = True)
airlines


# In[200]:


airlines.hist()


# In[201]:


airlines.plot(kind='kde')


# In[202]:


airlines = pd.read_excel("Airlines+Data.xlsx",index_col = 0,header = 0,parse_dates = True,squeeze=True)
airlines


# In[203]:


type(airlines)


# In[204]:


groups = airlines.groupby(pd.Grouper(freq='A'))
groups


# In[205]:


years = pd.DataFrame()

for name, group in groups:
    years[name.year] = group.values

years


# In[206]:


plt.figure(figsize = (15,7))
years.boxplot()


# In[207]:


plt.figure(figsize = (15,9))
pd.plotting.lag_plot(airlines)


# In[208]:


from statsmodels.graphics.tsaplots import plot_acf

plt.figure(figsize = (32,20))
plot_acf(airlines, lags=95)
plt.show()


# In[209]:


airlines = pd.read_excel("Airlines+Data.xlsx",index_col = 0,header = 0,parse_dates = True,squeeze=True)
airlines


# In[210]:


airlines.shape


# In[211]:


upsampled = airlines.resample('D').mean()
upsampled.head(20)


# In[212]:


upsampled.shape


# In[213]:


interpolated = upsampled.interpolate(method='linear')
interpolated.head(30)


# In[214]:


airlines.plot()


# In[215]:


resample = airlines.resample('Q')
quarterly_mean_sales = resample.mean()


# In[216]:


quarterly_mean_sales.plot()


# In[217]:


airlines=pd.read_excel("Airlines+Data.xlsx",index_col=0,header=0, parse_dates=True)
airlines


# In[218]:


# line plot
plt.subplot(211)
plt.plot(airlines)


# In[219]:


# histogram
plt.subplot(212)
plt.hist(airlines)


# In[220]:


dataframe = pd.DataFrame(np.log(airlines.values), columns = ['Passengers'])
dataframe


# In[221]:


# line plot
plt.subplot(211)
plt.plot(dataframe['Passengers'])


# In[222]:


# histogram
plt.subplot(212)
plt.hist(dataframe['Passengers'])
plt.show()


# In[223]:


quarterly_mean_sales.head()


# In[224]:


dataframe = pd.DataFrame(np.sqrt(airlines.values), columns = ['Passengers'])
dataframe


# In[225]:


# line plot
plt.subplot(211)
plt.plot(dataframe['Passengers'])


# In[226]:


# histogram
plt.subplot(212)
plt.hist(dataframe['Passengers'])
plt.show()


# In[227]:


airlines=pd.read_excel("Airlines+Data.xlsx")
airlines


# In[228]:


airlines['Passengers'].plot()


# In[229]:


airlines["month"] = airlines['Month'].dt.strftime("%b")
airlines["year"] = airlines['Month'].dt.strftime("%Y")


# In[230]:


airlines


# In[231]:


mp = pd.pivot_table(data = airlines,values = "Passengers",index = "year",columns = "month",aggfunc = "mean",fill_value=0)
mp


# In[232]:


plt.figure(figsize=(12,8))
sns.heatmap(mp,annot=True,fmt="g",cmap = 'YlGnBu')


# In[233]:


plt.figure(figsize=(15,10))

plt.subplot(211)
sns.boxplot(x="month",y="Passengers",data=airlines)

plt.subplot(212)
sns.boxplot(x="year",y="Passengers",data=airlines)


# In[234]:


plt.figure(figsize=(17,8))
sns.lineplot(x="year",y="Passengers",data=airlines)


# In[235]:


airlines


# In[236]:


airlines.shape


# In[237]:


airlines['t']=np.arange(1,97)
airlines['t_square']=np.square(airlines.t)
airlines['log_Passengers']=np.log(airlines.Passengers)
airlines2=pd.get_dummies(airlines['month'])


# In[238]:


airlines


# In[239]:


airlines2


# In[240]:


airlines=pd.concat([airlines,airlines2],axis=1)
airlines


# In[241]:


Train = airlines.head(84)
Test = airlines.tail(12)


# In[242]:


Train


# In[243]:


Test


# In[244]:


#Linear Model
import statsmodels.formula.api as smf 

linear_model = smf.ols('Passengers~t',data=Train).fit()
pred_linear =  pd.Series(linear_model.predict(pd.DataFrame(Test['t'])))
rmse_linear = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_linear))**2))
rmse_linear


# In[245]:


#Exponential
Exp = smf.ols('log_Passengers~t',data=Train).fit()
pred_Exp = pd.Series(Exp.predict(pd.DataFrame(Test['t'])))
rmse_Exp = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Exp)))**2))
rmse_Exp


# In[246]:


#Quadratic 
Quad = smf.ols('Passengers~t+t_square',data=Train).fit()
pred_Quad = pd.Series(Quad.predict(Test[["t","t_square"]]))
rmse_Quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_Quad))**2))
rmse_Quad


# In[247]:


#Additive seasonality 
add_sea = smf.ols('Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea = pd.Series(add_sea.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov']]))
rmse_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea))**2))
rmse_add_sea


# In[248]:


#Additive Seasonality Quadratic 
add_sea_Quad = smf.ols('Passengers~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_add_sea_quad = pd.Series(add_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
rmse_add_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_add_sea_quad))**2))
rmse_add_sea_quad


# In[249]:


##Multiplicative Seasonality
Mul_sea = smf.ols('log_Passengers~Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_sea = pd.Series(Mul_sea.predict(Test))
rmse_Mult_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_sea)))**2))
rmse_Mult_sea


# In[250]:


#Multiplicative Additive Seasonality 
Mul_Add_sea = smf.ols('log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data = Train).fit()
pred_Mult_add_sea = pd.Series(Mul_Add_sea.predict(Test))
rmse_Mult_add_sea = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(np.exp(pred_Mult_add_sea)))**2))
rmse_Mult_add_sea


# In[251]:


#Multiplicative Seasonality Quadratic 
mult_sea_Quad = smf.ols('log_Passengers~t+t_square+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=Train).fit()
pred_mult_sea_quad = pd.Series(mult_sea_Quad.predict(Test[['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','t','t_square']]))
rmse_mult_sea_quad = np.sqrt(np.mean((np.array(Test['Passengers'])-np.array(pred_mult_sea_quad))**2))
rmse_mult_sea_quad


# In[252]:


data = {"MODEL":pd.Series(["rmse_linear","rmse_Exp","rmse_Quad","rmse_add_sea","rmse_add_sea_quad","rmse_Mult_sea","rmse_Mult_add_sea"]),"RMSE_Values":pd.Series([rmse_linear,rmse_Exp,rmse_Quad,rmse_add_sea,rmse_add_sea_quad,rmse_Mult_sea,rmse_Mult_add_sea])}
table_rmse=pd.DataFrame(data)
table_rmse.sort_values(['RMSE_Values'])


# In[253]:


model_full = smf.ols('log_Passengers~t+Jan+Feb+Mar+Apr+May+Jun+Jul+Aug+Sep+Oct+Nov',data=airlines).fit()


# In[254]:


pred_new  = pd.Series(model_full.predict(airlines))
pred_new


# In[255]:


airlines["forecasted_Passengers"] = pd.Series(np.exp(pred_new))


# In[256]:


plt.figure(figsize=(15,10))
plt.plot(airlines[['Passengers','forecasted_Passengers']].reset_index(drop=True))


# In[257]:


airlines=pd.read_excel("Airlines+Data.xlsx")
Train = airlines.head(84)
Test = airlines.tail(12)


# In[258]:


Train


# In[259]:


Test


# In[260]:


plt.figure(figsize=(24,7))
airlines['Passengers'].plot(label="org")
airlines["Passengers"].rolling(15).mean().plot(label=str(5))
plt.legend(loc='best')


# In[261]:


plt.figure(figsize=(24,7))
airlines['Passengers'].plot(label="org")
for i in range(2,24,6):
    airlines["Passengers"].rolling(i).mean().plot(label=str(i))
plt.legend(loc='best')


# In[262]:


decompose_ts_add = seasonal_decompose(airlines['Passengers'], period = 12)
decompose_ts_add.plot()
plt.show()


# In[263]:


import statsmodels.graphics.tsaplots as tsa_plots
tsa_plots.plot_acf(airlines.Passengers,lags=12)
tsa_plots.plot_pacf(airlines.Passengers,lags=12)
plt.show()


# In[264]:


def MAPE(pred,org):
    temp = np.abs((pred-org)/org)*100
    return np.mean(temp)


# In[265]:


ses_model = SimpleExpSmoothing(Train["Passengers"]).fit(smoothing_level=0.2)
pred_ses = ses_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_ses,Test.Passengers)


# In[266]:


hw_model = Holt(Train["Passengers"]).fit(smoothing_level=0.8, smoothing_slope=0.2)
pred_hw = hw_model.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hw,Test.Passengers) 


# In[267]:


hwe_model_add_add = ExponentialSmoothing(Train["Passengers"],seasonal="add",trend="add",seasonal_periods=12).fit() #add the trend to the model
pred_hwe_add_add = hwe_model_add_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_add_add,Test.Passengers) 


# In[268]:


hwe_model_mul_add = ExponentialSmoothing(Train["Passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit() 
pred_hwe_mul_add = hwe_model_mul_add.predict(start = Test.index[0],end = Test.index[-1])
MAPE(pred_hwe_mul_add,Test.Passengers)


# In[269]:


hwe_model_mul_add = ExponentialSmoothing(airlines["Passengers"],seasonal="mul",trend="add",seasonal_periods=12).fit() 


# In[270]:


hwe_model_mul_add.forecast(12)


# In[ ]:




