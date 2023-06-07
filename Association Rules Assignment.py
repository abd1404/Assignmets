#!/usr/bin/env python
# coding: utf-8

# Prepare rules for the all the data sets 
# 1) Try different values of support and confidence. Observe the change in number of rules for different support,confidence values
# 2) Change the minimum length in apriori algorithm
# 3) Visulize the obtained rules using different plots 

# In[11]:


import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[12]:


book=pd.read_csv('book.csv')
book


# In[13]:


df=pd.get_dummies(book)
df


# In[14]:


from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder


# In[10]:


frequent_itemsets = apriori(df, min_support = 0.1, use_colnames = True)
frequent_itemsets


# In[15]:


rules = association_rules(frequent_itemsets, metric = 'lift', min_threshold = 0.7)
rules


# In[16]:


rules.sort_values('lift',ascending= False)


# In[17]:


rules[rules.lift>1]


# In[18]:


rules[rules.lift<1]


# In[19]:


rules[rules.leverage == 0]


# In[20]:


frequent_itemset = apriori(df,min_support = 0.2, use_colnames = True)
rules = association_rules(frequent_itemset, metric = 'lift', min_threshold = 0.9)
rules


# In[21]:


frequent_itemset = apriori(df, min_support = 0.03, use_colnames = True)
rules = association_rules(frequent_itemset, metric = 'lift', min_threshold = 0.6)
rules


# In[22]:


import matplotlib.pyplot as plt
plt.scatter(rules.support,rules.confidence)
plt.xlabel('Support')
plt.ylabel('Confidence')


# In[23]:


frequent_itemset = apriori(df, min_support = 0.23, use_colnames = True)
rules = association_rules(frequent_itemset, metric = 'lift', min_threshold = 0.7)
rules


# In[24]:


import matplotlib.pyplot as plt
plt.scatter(rules.support,rules.confidence)
plt.xlabel('Support')
plt.ylabel('Confidence')


# In[25]:


frequent_itemset = apriori(df, min_support = 0.13, use_colnames = True)
rules = association_rules(frequent_itemset, metric = 'lift', min_threshold = 0.6)
rules


# In[26]:


import matplotlib.pyplot as plt
plt.scatter(rules.support,rules.confidence)
plt.xlabel('Support')
plt.ylabel('Confidence')


# In[27]:


frequent_itemset1 = apriori(df, min_support = 0.05, use_colnames = True)
rules = association_rules(frequent_itemset1, metric = 'lift', min_threshold = 0.9)
rules


# In[28]:


import matplotlib.pyplot as plt
plt.scatter(rules.support,rules.confidence)
plt.xlabel('Support')
plt.ylabel('Confidence')


# In[ ]:





# 2) 

# In[29]:


movies = pd.read_csv('my_movies.csv')
movies


# In[30]:


movies.shape


# In[31]:


movies.describe()


# In[32]:


movies.corr()


# In[33]:


movies.info()


# In[34]:


df = movies.iloc[:,5:]
df


# In[35]:


frequent_itemset = apriori(df, min_support = 0.02, use_colnames = True)
frequent_itemset


# In[36]:


rules = association_rules(frequent_itemset, metric = 'lift', min_threshold = 0.7)
rules


# In[37]:


plt.scatter(rules.support, rules.confidence)
plt.xlabel('Support')
plt.ylabel('Confidence')


# In[38]:


frequent_itemset = apriori(df, min_support = 0.05, use_colnames = True)
frequent_itemset


# In[39]:


rules = association_rules(frequent_itemset, metric ='lift', min_threshold = 0.9)
rules


# In[40]:


import matplotlib.pyplot as plt
plt.scatter(rules.support,rules.confidence)
plt.xlabel('Support')
plt.ylabel('Confidence')


# In[41]:


frequent_itemset1 = apriori(df, min_support = 0.1, use_colnames = True)
frequent_itemset1


# In[42]:


rules = association_rules(frequent_itemset1, metric = 'lift', min_threshold = 0.95)
rules


# In[43]:


import matplotlib.pyplot as plt
plt.scatter(rules.support,rules.confidence)
plt.xlabel('Support')
plt.ylabel('Confidence')


# In[ ]:




