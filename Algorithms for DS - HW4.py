#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Roger H Hayden III
#Algorithms for DS
#Homework 4


# In[2]:


#Importing packages
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import math
from sklearn.metrics import accuracy_score

from numpy import loadtxt
from keras.models import Sequential
from keras.layers import Dense


# In[3]:


#Read in Data
df = pd.read_csv(r'C:\Users\roger\OneDrive\Desktop\iris.csv')
print(df)


# In[4]:


df.head()


# In[5]:


cols = ["sepal_length","sepal_width","petal_length","petal_width","species"]
df_encode = df.drop(cols, axis= 1)
df_encode = df_encode.apply(LabelEncoder().fit_transform)
df_rest = df[cols]
df = pd.concat([df_rest,df_encode], axis= 1)


# In[6]:


df


# In[7]:


X= df.iloc[:,:4].values

spread = 0.14


# In[8]:


X


# In[9]:


class RBNmodel:
    def __init__(self, W_hat, W, spread, pred_error):
        self.W_hat = W_hat
        self.W = W
        self.spread = spread
        self.pred_error = pred_error


# def multiDiag(X1, X2):
#     r1, c1 = X1.shape
#     r2, c2 = X2.shape
#     
#     X1temp = np.transpose(X1)
#     
#     X = np.zeros((c1, r1))
#     X = X1temp*X2
#     r1,c1 = X.shape
#     
#     if r1 > 1:
#         XDiag = np.transpose(np.sum(X, axis = 0))
#         return XDiag
#     else:
#         return np.transpose(X)

# In[10]:


def multiDiag(X1, X2):
    r1, c1 = X1.shape
    r2, c2 = X2.shape
    
    X1temp = np.transpose(X1)
    X = np.zeros((c1, r1),dtype = int)
    X = X1temp * X2
    r1, c1 = X.shape
    if r1 > 1:
        xDiag = np.transpose(np.sum(X, axis = 0))
        return xDiag
    else:
        return np.transpose(X)


# # RBF Train No Bias Algorithm Analysis
# 
# 
# 

# The RBF Train NoBias Function is O(n) since there is one loop contained in the function
# 
# 

# The RBF Train NoBias function T(n) also appears to be n

# In[11]:


def RBF_Train_NoBias(X, y, spread = 0.5):
    n,d = X.shape
    
    X = np.transpose(X)
    
    H = np.zeros((n, n))
    
    for j in range(0, n):
        W = X[:, j].reshape(d, 1)
        
        repeat_matrix = np.repeat(W, n, axis = 1)
        
        D = X - repeat_matrix
        s = multiDiag(np.transpose(D),D)/(2*np.square(spread))
        s_neg = -1*s
        H[:, j] = np.exp(s_neg)
    
    W_hat = np.linalg.pinv(np.transpose(H) @ H) @ np.transpose(H) @ y
    #W_hat = np.linalg.pinv(H)*y
    yt = np.transpose(H @ W_hat)
    ypred = np.ones((y.shape))
    neg_value_idx = np.where(yt < 0)
    ypred[neg_value_idx] = -1
    predError = 1 - len(y == ypred)/len(y)
    
    return RBNmodel(W_hat, X, spread, predError)


# In[12]:


#def RBF_Train_NoBias(X, y, spread = 0.5):
#    n,d = X.shape
#    
#    X = np.transpose(X)
#    
#    H = np.zeros((n, n))
#    
#    for j in range(0, n):
#        W = X[:, j].reshape(d, 1)
#        
#        repeat_matrix = np.repeat(W, n, axis = 1)
#        
#        D = X - repeat_matrix
#        s = multiDiag(np.transpose(D),D)/(2*np.square(spread))
#        s_neg = -1*s
#        H[:, j] = np.exp(s_neg)
#    
#    W_hat = np.linalg.pinv(np.transpose(H) @ H)@ np.transpose(H) @ y
#    yt = np.transpose(H@W_hat)
#    ypred = np.ones((y.shape))
#    index_negative = np.where(yt < 0)
#    ypred[index_negative] = -1
#    predError = 1- len(y == ypred)/ len(y)
#    return RbnModel(W_hat, X, spread,predError)


# In[13]:


def RBF_Classify_NoBias(X, model):
    n1, d1 = X.shape
    x = np.transpose(X)
    
    n2, d2 = np.transpose(model.W).shape
    
    H = np.transpose(X)
    #H = np.zeros((n1, n2))
    for j in range(0, n2):
        M = model.W[:,j].reshape(d2, 1)
        
        repeat_matrix = np.repeat(M, n1, axis = 1)
        D = X - repeat_matrix
        
        s = multiDiag(np.transpose(D),D) / (2*np.square(model.spread))
        s_neg = -1*s
        H[:,j] = np.exp(s_neg)
        
    y = np.transpose(H @ model.W_hat)
    ypred = np.ones((y.shape))
    neg_value_idx = np.where(y < 0)
    ypred[neg_value_idx] = -1
    
    return (y, ypred)


# In[14]:


Y_1 = np.concatenate((np.ones((50, 1)), np.ones((50, 1))*-1, np.ones((50, 1))*-1), axis = 0).flatten()
model1 = RBF_Train_NoBias(X, Y_1, spread)
Y_2 = np.concatenate((np.ones((50, 1))*-1, np.ones((50, 1)), np.ones((50, 1))*-1), axis = 0).flatten()
model2 = RBF_Train_NoBias(X, Y_2, spread)
Y_3 = np.concatenate((np.ones((50, 1))*-1, np.ones((50, 1))*-1, np.ones((50, 1))), axis = 0).flatten()
model3 = RBF_Train_NoBias(X, Y_3, spread)


# In[15]:


x0 = np.array([5.1, 3.5, 1.4, 0.2])
x0_len = len(x0)
x0 = x0.reshape(1, x0_len)

x0


# In[16]:


#Hitting the error here with the RBF Classify No Bias Function
yt0_1, ypred0_1 = RBF_Classify_NoBias(x0, model1)
yt0_2, ypred0_2 = RBF_Classify_NoBias(x0, model2)
yt0_3, ypred0_3 = RBF_Classify_NoBias(x0, model3)
tmp = np.array((yt0_1, yt0_2, yt0_3))
value = np.max(tmp)
y0pred = np.argmax(tmp)


# In[ ]:


yt1, ypred1 = RBF_Classify_NoBias(X, model1)
yt2, ypred2 = RBF_Classify_NoBias(X, model2)
yt3, ypred3 = RBF_Classify_NoBias(X, model3)
tmp = np.array([yt1, yt2, yt3])
value = np.amax(tmp, axis = 0)
ypred = np.argmax(tmp, axis = 0) + 1


# In[ ]:


Y = np.concatenate((np.ones((50, 1)), np.ones((50, 1))*2, np.ones((50, 1))*3), axis = 0).flatten()
accuracy = (len(Y == ypred)/150)*100
prnint(accuracy)

