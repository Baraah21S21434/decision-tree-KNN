#!/usr/bin/env python
# coding: utf-8

# In[4]:


import pandas as pd
import numpy as np
import seaborn as sns
import sklearn.svm as svm


# In[5]:


dataset = pd.read_csv(r'C:\Users\HP\Desktop\car_data.csv')


# In[6]:


dataset


# In[7]:


#Let's count the number of null values in each column.

dataset.isnull().sum()


# In[ ]:




