#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# In[2]:


#Import the dataset
import sklearn.svm as svm


# In[3]:


dataset = pd.read_csv(r'C:\Users\HP\Desktop\car_data.csv')
dataset.head()


# In[4]:


dataset.info()


# In[5]:


dataset['Gender'].value_counts()


# In[6]:


#Converting gender values from object values to numerical values
#A sign Female to (0) and Male to (1) 

convert = {"Gender": {"Female":0, "Male":1}}


# In[7]:


dataset = dataset.replace(convert)


# In[8]:


#dataset after convert the gender to numerical values
#data analysis
dataset


# In[9]:


from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
dataset['Purchased'] = le.fit_transform(dataset['Purchased'])
dataset.head(100)


# In[10]:


plt.figure(1)
sns.heatmap(dataset.corr())
plt.title('Correlation On Purchased Car Classes')


# In[11]:


#Execute the following code to split the data into training and test sets
from sklearn.model_selection import train_test_split


# In[13]:


#Data processing
X = dataset.drop(columns = ['Purchased'])
Y = dataset['Purchased']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.25)


# In[14]:


#Training and making predictions
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
classifier = DecisionTreeClassifier()
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(X_test)


# In[15]:


# Summary of the predictions made by the classifier
print(classification_report(Y_test, y_pred))
print(confusion_matrix(Y_test, y_pred))
# Accuracy score
from sklearn.metrics import accuracy_score
print('Accuracy is',accuracy_score(y_pred,Y_test))


# In[ ]:




