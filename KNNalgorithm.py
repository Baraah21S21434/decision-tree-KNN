#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


# In[2]:


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
dataset


# In[9]:


X = dataset.iloc[:, :-1].values 
y = dataset.iloc[:, 4].values


# In[10]:


from sklearn.model_selection import train_test_split 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)


# In[11]:


from sklearn.preprocessing import StandardScaler 
scaler = StandardScaler() 
scaler.fit(X_train) 
X_train = scaler.transform(X_train) 
X_test = scaler.transform(X_test)


# In[12]:


from sklearn.neighbors import KNeighborsClassifier 
classifier = KNeighborsClassifier(n_neighbors=5) 
classifier.fit(X_train, y_train)


# In[13]:


y_pred = classifier.predict(X_test)


# In[14]:


from sklearn.metrics import classification_report, confusion_matrix 
print(confusion_matrix(y_test, y_pred)) 
print(classification_report(y_test, y_pred))


# In[15]:


# Comparing the error values with the change in k values - the number of neighbors
#- by calculating the error value for k values between 1 and 40 
#drawing the graph

error = [] 
# Calculating error for K values between 1 and 40 
for i in range(1, 40): 
   knn = KNeighborsClassifier(n_neighbors=i) 
   knn.fit(X_train, y_train) 
   pred_i = knn.predict(X_test) 
   error.append(np.mean(pred_i != y_test))


# In[17]:


plt.figure(figsize=(12, 6)) 
plt.plot(range(1, 40), error, 
   color='red', linestyle='dashed', 
   marker='o', markerfacecolor='blue', 
   markersize=10) 
plt.title('Error Rate K Value') 
#plt.xlabel('K Value') plt.ylabel('Mean Error')


# In[ ]:




