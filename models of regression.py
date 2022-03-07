#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from matplotlib import pyplot
from sklearn.metrics import mean_squared_error

from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge, Lasso

from tabulate import tabulate


# In[2]:


# generate regression dataset
X, Y = make_regression(n_samples=500, n_features=15,random_state=10)
#split 
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=.3)


# In[3]:


#linear regression
linear_r = LinearRegression()
linear_r.fit(X_train, Y_train)
Y_pred_ = linear_r.predict(X_train)

#lasso regression
lasso_r = Lasso(alpha=0.0001)
lasso_r = lasso_r.fit(X_train,Y_train)
Y_pred_lr = lasso_r.predict(X_train)

#ridge regression
ridge_r = Ridge(alpha=0.001)
ridge_r = ridge_r.fit(X_train,Y_train)
Y_pred_rr = ridge_r.predict(X_train)


#Polynomial regression

poly_f = PolynomialFeatures(degree=2)
l_r = LinearRegression()

X_poly = poly_f.fit_transform(X_train)
l_r = l_r.fit(X_poly, Y_train)
Y_pred_p = l_r.predict(X_poly)


# In[4]:


ml=mean_squared_error(Y_pred_, Y_train)
mls=mean_squared_error(Y_pred_lr, Y_train)
mr=mean_squared_error(Y_pred_rr, Y_train)
mp=mean_squared_error(Y_pred_p, Y_train)


# In[5]:


table = [["linear regression",ml],["lasso",mls], ["ridge",mr],["polynominal",mp]]
print(tabulate(table,headers=["model","MSE"]))


# In[6]:


#linear test
tY_pred_ = linear_r.predict(X_test)

#lasso test
tY_pred_lr = lasso_r.predict(X_test)


#ridge test
tY_pred_rr = ridge_r.predict(X_test)

#Polynomial test
poly_f = PolynomialFeatures(2)
l_r = LinearRegression()


X_poly = poly_f.fit_transform(X_test)
l_r = l_r.fit(X_poly, Y_test)
tY_pred_p = l_r.predict(X_poly)


# In[7]:


ml_t=mean_squared_error(tY_pred_, Y_test)
mls_t=mean_squared_error(tY_pred_lr, Y_test)
mr_t=mean_squared_error(tY_pred_rr, Y_test)
mp_t=mean_squared_error(tY_pred_p, Y_test)


# In[8]:


table = [["linear regression",ml_t],["lasso",mls_t], ["ridge",mr_t],["polynominal",mp_t]]
print(tabulate(table,headers=["model","MSE"]))


# In[ ]:




