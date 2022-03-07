#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


# In[13]:


df = pd.read_csv('dataset_q2_q4.csv'  )
df.head()


# In[14]:


#split 
X = df[['x1',"x2","x3"]].values

y1 = df['y_label'].values
m=len(y1)


# In[15]:


#add column
Xx = np.append(np.ones([m,1]), X, axis=1) 


# In[16]:


x_tr,x_ts,y_tr,y_ts=train_test_split(Xx,y1,test_size=.4)


# In[17]:


theta_n= np.linalg.inv((x_tr.T.dot(x_tr))).dot(x_tr.T.dot(y_tr))
print(theta_n)


# In[18]:


def predict(X,theta):
    theta0=theta[0]
    theta1=theta[1]
    theta2=theta[2]
    theta3=theta[3]

    x1=X[:,0]
    x2=X[:,1]
    x3=X[:,2]
    
    pred=3*theta1*(x1**2)+theta2*(x2**3)-theta3*(x2**3)+theta2*theta3*(x3**3)+theta0

    return pred


# In[19]:


l=predict(x_ts,theta_n)


# In[20]:


mse1=mean_squared_error(y_ts,l)
mse1


# In[ ]:




