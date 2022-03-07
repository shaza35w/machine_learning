#!/usr/bin/env python
# coding: utf-8

# In[12]:


from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn import linear_model
from sklearn.metrics import mean_squared_error


# In[13]:


import numpy as np
import pandas as pd


# ![Screen%20Shot%202022-02-26%20at%207.06.47%20PM.png](attachment:Screen%20Shot%202022-02-26%20at%207.06.47%20PM.png)

# In[ ]:





# In[14]:


dataset=pd.read_csv('dataset_q2_q4.csv')


# In[15]:


dataset


# In[16]:


y=dataset['y_label']
X=dataset.drop('y_label', axis=1)


# In[ ]:





# In[17]:


print(X.shape)
print(y.shape)


# In[18]:


X.head()


# In[19]:


X=np.array(X)
y=np.array(y)


# In[20]:


x_train,x_test,y_train,y_test=train_test_split(X,y,test_size=.4)

print(x_train.shape)
print(x_test.shape)


# In[21]:



def fit_line(X,Y,lr=0.000000000000000000000002,epochs=80000):
    
    theta0=0
    theta1=0
    theta2=0
    theta3=0
    
    x1=X[:,0]
    x2=X[:,1]
    x3=X[:,2]
    
    losses=[]
    
    for i in range(epochs):
        
        y_hat=3*theta1*(x1**2)+theta2*(x3**3)-theta3*(x2**3)+theta2*theta3*(x3**3)+theta0
        cost=(sum((y_hat-y_train)**2))/(2*y_train.size)
        losses.append(cost)
        
        grad0=sum(y_hat-y_train)/(y_train.size)
        grad1=np.matmul(3*(x1**3),(y_hat-y_train)/(y_train.size))
        grad2=np.matmul(((x2**3)+theta3*(x3**3)),(y_hat-y_train)/(y_train.size))
        grad3=np.matmul(((-x2**3)+theta2*(x3**3)),(y_hat-y_train)/(y_train.size))
        
        theta0=theta0-lr*grad0
        theta1=theta1-lr*grad1
        theta2=theta2-lr*grad2
        theta3=theta3-lr*grad3
        
        
    plt.plot(list(range(epochs)),losses) 
    
    
    print('thetas: ',theta0,theta1,theta2,theta3)
    print(y_hat[87],y_hat[0])
    print('loss',cost)
    
    return theta0,theta1,theta2,theta3


# In[30]:


theta0,theta1,theta2,theta3=fit_line(x_train,y_train,lr=0.0000000000000000000002,epochs=800000)


# In[37]:


def preduct(X,th0,th1,th2,th3):
    
    x1=X[:,0]
    x2=X[:,1]
    x3=X[:,2]
    y_hat_test=3*theta1*(x1**2)+theta2*(x2**3)-theta3*(x2**3)+theta2*theta3*(x3**3)+theta0


    return y_hat_test


# In[38]:


xt=preduct(x_test,theta0,theta1,theta2,theta3)


# In[39]:


mse=mean_squared_error(y_test,xt)
mse


# In[ ]:




