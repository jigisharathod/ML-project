#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_csv(r"C:\Users\Dev\Downloads\Complete-Python-3-Bootcamp-master/OnlineNewsPopularity.csv")
df.head()


# In[2]:


from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import LinearRegression
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split


# In[3]:


df.columns


# In[26]:


df.columns = df.columns.str.replace(' ', '')


# In[27]:


df.columns


# In[6]:


df.shape # here we have 39644 rows and 61 columns


# In[28]:


df.describe()


# In[ ]:





# In[30]:


df.isnull().any()


# In[31]:


df.dtypes


# In[32]:


df =df.drop(['weekday_is_monday', 'weekday_is_tuesday', 'weekday_is_wednesday', 'weekday_is_thursday', 'weekday_is_friday', 'weekday_is_saturday', 'weekday_is_sunday'   
  ], axis=1)


# In[33]:


df.shape


# In[34]:


df.head(5)


# In[35]:


df.columns


# In[36]:


plt.hist(df['shares'])
plt.show()


# In[37]:



plt.hist(np.log10(df['shares']))
plt.show()


# In[38]:


np.log10(df['shares']).describe()


# In[39]:



#Separating URLs from the main data
df = df.drop(['url'],axis=1)


# In[40]:


for i in df.columns:
    sns.boxplot(df[i])
    plt.show()


# In[41]:


#removingoutliers
Q1 = df.quantile(q=0.25) 

Q3 = df.quantile(q=0.75)

IQR = Q3-Q1
print('IQR for each column:- ')
print(IQR)


# In[42]:


df.columns


# In[43]:


df['shares'] = (df['shares'] > 1400).astype(int) # Popular =1 , Unpopular = 0
df


# In[44]:


df['shares'].head(5)


# In[50]:


df.columns


# In[58]:


df['n_non_stop_words'].unique()


# In[60]:


df = df.drop(['n_non_stop_words'],axis=1)


# In[61]:


df.columns


# In[62]:


sns.set_style('whitegrid')
sns.countplot(x='shares',data=df,palette='RdBu_r')


# In[63]:


sns.distplot(df['average_token_length'].dropna(),kde=False,color='darkred',bins=30)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




