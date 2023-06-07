#!/usr/bin/env python
# coding: utf-8

# 1) A F&B manager wants to determine whether there is any significant difference in the diameter of the cutlet between two units. A randomly selected sample of cutlets was collected from both units and measured? Analyze the data and draw inferences at 5% significance level. Please state the assumptions and tests that you carried out to check validity of the assumptions.
# 
# 
#      Minitab File : Cutlets.mtw
# ![image.png](attachment:image.png)

# In[1]:


import numpy as np
import pandas as pd

df=pd.read_csv('Cutlets.csv')


# In[2]:


df


# H0:To prove there is a significant difference
# 
# H1:To prove there is not a significant difference

# In[3]:


df['Unit A'].mean()


# In[4]:


df['Unit B'].mean()


# In[5]:


from scipy import stats


# In[6]:


Zcal,pval=stats.ttest_ind(df['Unit A'],df['Unit B'])
print('Z calculated value is',Zcal.round(4))
print('P value is:',pval.round(4))


# In[8]:


if pval<0.05:
    print('Reject null hypothesis and accept alternative hypothesis')
else:
    print('Reject alternative hypothesis and accept null hypothesis')


# P value comes under the accepted region. Hence, there is no significant difference between Unit A and Unit B.

# In[ ]:





# 2) A hospital wants to determine whether there is any difference in the average Turn Around Time (TAT) of reports of the laboratories on their preferred list. They collected a random sample and recorded TAT for reports of 4 laboratories. TAT is defined as sample collected to report dispatch.
#    
#   Analyze the data and determine whether there is any difference in average TAT among the different laboratories at 5% significance level.
#  
#  
#     Minitab File: LabTAT.mtw
# 

# H0: There is no significant difference
# 
# H1: There is atleast one significant difference

# In[9]:


df=pd.read_csv('LabTAT.csv')
df


# In[10]:


import pandas as pd
df=pd.read_csv('LabTAT.csv')
df
from scipy import stats
zcal, pval,f,jk=stats.chi2_contingency(df)
print(zcal, pval)
if pval < 0.05:
    print('Reject Null Hypothesis and Accept Alternative Hypothesis')
    print("the defective % varies by error")
else:
    print("Accept Null Hypothesis and Reject Alternative Hypothesis")
    print("the defective % doesn't varies by error")


# Z value falls under accepted region. So, there is a significant difference.

# In[ ]:





# 3) Sales of products in four different regions is tabulated for males and females. Find if male-female buyer rations are similar across regions.

# In[11]:


df=pd.read_csv('BuyerRatio.csv')
df


# H0: All proportions are equal
#     
# H1: Not all proportions are equal

# In[15]:


import researchpy as rp


# In[16]:


table,pvalue=rp.crosstab(df['East'],df['West'],test='chi-square')


# In[18]:


pvalue


# In[19]:


pvalue=0.1573
if pvalue < 0.05:
    print("Reject Null Hypothesis and Accept Alternative Hypothesis\n All proportions are not equal")
else:
    print("Accept Null Hypothesis and Reject Alternative Hypothesis \n All proportions are equal")


# In[20]:


table,pvalue=rp.crosstab(df['East'],df['North'],test='chi-square')
pvalue


# In[21]:


pvalue=0.1573
if pvalue < 0.05:
    print("Reject Null Hypothesis and Accept Alternative Hypothesis\n All proportions are not equal")
else:
    print("Accept Null Hypothesis and Reject Alternative Hypothesis \n All proportions are equal")


# In[22]:


table,pvalue=rp.crosstab(df['East'],df['South'],test='chi-square')
pvalue


# In[23]:


pvalue=0.1573
if pvalue < 0.05:
    print("Reject Null Hypothesis and Accept Alternative Hypothesis\n All proportions are not equal")
else:
    print("Accept Null Hypothesis and Reject Alternative Hypothesis \n All proportions are equal")


# In[24]:


table,pvalue=rp.crosstab(df['West'],df['North'],test='chi-square')
pvalue


# In[25]:


pvalue=0.1573
if pvalue < 0.05:
    print("Reject Null Hypothesis and Accept Alternative Hypothesis\n All proportions are not equal")
else:
    print("Accept Null Hypothesis and Reject Alternative Hypothesis \n All proportions are equal")


# In[26]:


table,pvalue=rp.crosstab(df['West'],df['South'],test='chi-square')
pvalue


# In[27]:


pvalue=0.1573
if pvalue < 0.05:
    print("Reject Null Hypothesis and Accept Alternative Hypothesis\n All proportions are not equal")
else:
    print("Accept Null Hypothesis and Reject Alternative Hypothesis \n All proportions are equal")


# In[28]:


table,pvalue=rp.crosstab(df['North'],df['South'],test='chi-square')
pvalue


# In[29]:


pvalue=0.1573
if pvalue < 0.05:
    print("Reject Null Hypothesis and Accept Alternative Hypothesis\n All proportions are not equal")
else:
    print("Accept Null Hypothesis and Reject Alternative Hypothesis \n All proportions are equal")


# P value comes under accepted region. Hence, all proportions are equal

# In[ ]:





# 4) TeleCall uses 4 centers around the globe to process customer order forms. They audit a certain %  of the customer order forms. Any error in order form renders it defective and has to be reworked before processing.  The manager wants to check whether the defective %  varies by centre. Please analyze the data at 5% significance level and help the manager draw appropriate inferences
# 
# Minitab File: CustomerOrderForm.mtw
#  
# 

# In[37]:


df=pd.read_csv('Costomer+OrderForm.csv')
df


# Ho: There is no significant difference
#     
# H1: There is significant difference

# In[43]:


phillippines=pd.DataFrame(df['Phillippines'].value_counts())
phillippines


# In[44]:


Indonesia=pd.DataFrame(df['Indonesia'].value_counts())
Indonesia


# In[45]:


India=pd.DataFrame(df['India'].value_counts())
India


# In[46]:


Malta=pd.DataFrame(df['Malta'].value_counts())
Malta


# In[48]:


import numpy as np
data=np.array([[280,267,271,269],[20,33,29,31]])
error_free=data[0]
defective=data[1]


# In[50]:


from scipy import stats
zcal, pval,f,jk=stats.chi2_contingency(data)
print(zcal, pval)
if pval < 0.05:
    print('Reject Null Hypothesis and Accept Alternative Hypothesis')
    print("the defective % varies by error")
else:
    print("Accept Null Hypothesis and Reject Alternative Hypothesis")
    print("the defective % doesn't varies by error")


# P value falls under accepted region. So, there is no significant variation

# In[ ]:




