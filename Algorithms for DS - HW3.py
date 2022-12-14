#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Roger H Hayden III
#Algorithms for DS
#Homework 3


# In[168]:


#Importing packages
import pandas as pd
import numpy as np
import statistics 
from scipy import stats
from statistics import mode
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.cluster import KMeans
from sklearn import cluster
from sklearn import preprocessing
from sklearn.mixture import GaussianMixture
from matplotlib import pyplot as plt
from PIL import Image


# In[107]:


#Read in Data
df = pd.read_csv(r'C:\Users\roger\OneDrive\Desktop\iris.csv')
print(df)


# In[108]:


df2 = pd.read_csv(r'C:\Users\roger\OneDrive\Desktop\Iris_Train.csv')
print(df2)


# PROBLEM 1
# 
# Followed examples in Office Hours. Tried to follow along and implement what was being done there to what I am doing. Ran into an error with creating the final table. The error was resorting back to the rank method. I attempted to find a solution however I am unsure how to fix this.

# In[159]:


def rank (class1, class2, n):
    meansq = pow(class1[n].mean() - class2[n].mean, 2)
    stdsq = pow(class1[n].std(), 2) + pow(class2[n].std(), 2)
    x = meansq/stdsq
    return x


# In[160]:


indices = []
index0 = df2.loc[df2['num'] == 0].drop(columns = ['num'])
index1 = df2.loc[df2['num'] == 1].drop(columns = ['num'])
index2 = df2.loc[df2['num'] == 2].drop(columns = ['num'])
index3 = df2.loc[df2['num'] == 3].drop(columns = ['num'])
index4 = df2.loc[df2['num'] == 4].drop(columns = ['num'])
index5 = df2.loc[df2['num'] == 5].drop(columns = ['num'])
index6 = df2.loc[df2['num'] == 6].drop(columns = ['num'])
index7 = df2.loc[df2['num'] == 7].drop(columns = ['num'])
index8 = df2.loc[df2['num'] == 8].drop(columns = ['num'])
index9 = df2.loc[df2['num'] == 9].drop(columns = ['num'])
indices = [index0, index1, index2, index3, index4, index5, index6, index7, index8, index9]


# In[161]:


print(indices)


# In[162]:


f_names = []
for col in df2.columns:
    if col == 'num':
        pass
    else:
        f_names.append(col)


# In[190]:


index_values = ['0','1','2','3','4','5','6','7','8','9']
rank_data = {}
cont = 0
test = []
for i in range(len(index_values)):
    for j in range(cont, len(index_values)):
        test = []
        if i == j:
            pass
        else:
            for k in range(len(f_names)):
                if k == 0:
                    test.append(rank(indices[i], indices[j], f_names[i]))
                else:
                    test.append(rank(indices[i], indices[j], f_names[i]))
            rank_data[indeces[i]+ 'vs' +indices[j]] = test
    cont = cont + 1


# In[ ]:


df3 = pd.DataFrame(rank_data)

df3.insert(0, "Features", f_names, allow_duplicates = True)

df3.to_csv('Problem1.csv', index = False)
print(df3)


# PROBLEM 2
# 
# Here I also tried to follow along with what was going on in Office Hours and implement the solutions here. In addition, I reached out to Sonny for assistance, however I keep running into an error here as well with the last loop before creating the plot.
# 
# I do believe the plot tells us that we did cluster the groups of each class rather well in this specific scenario. There is not too much overlap and it appears that the accuracy through much of these codes is meant to be high.

# In[19]:


em_data = df.drop(columns = ['sepal_length', 'sepal_width', 'species'])
em_data


# In[20]:


species = df['species']
species


# In[27]:


for i in range(len(species)):
    if species[i] == 'setosa':
        species[i] = 0
    elif species[i] == 'versicolor':
        species[i] = 1
    else:
        species[i] = 2


# In[28]:


print(species)


# In[182]:


kmeans = KMeans(n_clusters=3)
clusters = kmeans.fit_predict(em_data)


# In[183]:


scalar = preprocessing.StandardScaler()
scalar.fit(em_data)
scaled_X = scalar.transform(em_data)


# In[184]:


scaled = pd.DataFrame(scaled_X, columns = em_data.columns)


# In[185]:


plt.figure(figsize=(20, 20))

gmm = GaussianMixture(n_components=3)
gmm_y = gmm.fit_predict(scaled)


# In[186]:


print(gmm_y)


# In[187]:


labels = np.zeros_like(clusters)


# In[188]:


print(labels)


# In[193]:


for i in range(3):
  cat = (gmm_y == i)
  labels[cat] = mode(species[cat])[0]


# In[ ]:


acc = accuracy_score(species_list, labels)
print('Accuracy = ', acc)


# In[ ]:


plt.subplot(2, 2, 3)
plt.scatter(iris.petal_length, iris.petal_width, c = colormap[gmm_y], s=40)
plt.subplots_adjust(hspace=0.4, wspace=0.4)
plt.title('EM Clusters')
plt.xlabel('Petal Length')
plt.ylabel('Petal Width')


# PROBLEM 3
# 
# My Personal Definitions for each of the provided terms is in my report

# PROBLEM 4
# 
# I attempted to implement crossfold validation.
# 
# Each k-fold creates subsets of the data by chosing different subsets based off of the number of folds or splits we chose to use. In the case below we chose to use 10 and 5.

# Link used to help: https://www.statology.org/k-fold-cross-validation-in-python/?msclkid=e4772e15b07811ec99b4e3913d0e0386

# In[16]:


# Establish what you are predicting and the features
y = df[["sepal_length"]]
x = df[["sepal_width",
        "petal_length",
        "petal_width"]]
 
# train/test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=45)


# In[22]:


# Cross Validation Method
cv1 = KFold(n_splits=10, random_state=1, shuffle=True)

# Building Model as Linear Regression
iris_LR_model = LinearRegression()

# Use k-fold to look at the model
scores = cross_val_score(iris_LR_model, x, y, scoring='neg_mean_absolute_error',
                         cv=cv, n_jobs=-1)
# Mean Absolute Error (MAE) - The lower the value usually the better of a job the model does
np.mean(np.absolute(scores))


# In[25]:


# Cross Validation Method
cv = KFold(n_splits=5, random_state=1, shuffle=True) 

# Building Model as Linear Regression
iris_LR_model = LinearRegression()

# Use k-fold to look at the model
scores = cross_val_score(iris_LR_model, x, y, scoring='neg_mean_squared_error',
                         cv=cv, n_jobs=-1)

# Root Mean Squared Error (RMSE) - The lower value here usually means better as well
np.sqrt(np.mean(np.absolute(scores)))


# PROBLEM 5
# 
# Attempted to follow along in Office Hours and Implement a Similar Solution to what fellow classmates did. I was unsuccessful in creating a full solution to this problem. I believe the problem may have to do with how I am trying to use the data, however I am unsure.

# In[134]:


def GaussianKernel(X_test, X_train, spread):
    dim = len(X_test)
    obs = len(X_train)
    
    a = np.zeros(obs)
    
    for i in range(obs):
        a[i] = (1/obs)*(1/(np.sqrt(2*np.pi)*spread)**dim) * np.exp(-1*np.linalg.norm(X_test -X_train[i,:])**2/(2*spread**2))
        
    return a


# In[135]:


#O(n^2)
def ParzenWindow(df, space, spread):
    obs = len(space)
    spec_class = np.unique(df.species)
    pdf = np.zeros((len(spec_class), obs))
    
    i = 0
    
    for species in spec_class:
        j = np.array(df[df.species == species].loc[:, df.columns != 'species'])
        k = len(j)
        for l in range(k):
            pdf[i, :] += GaussianKernel(j[l, :], space, spread)
        
        pdf[i] / k
        
        i += 1
        
    return pdf


# In[136]:


spread = [0.1, 0.25, 0.5]


# In[137]:


One_Dim = np.reshape(np.arange(0, 8.05, 0.05), (-1, 1))


# In[138]:


Two_Dim = np.meshgrid(np.arange(0, 3.02, 0.05), np.arange(0, 7.05, 0.05))


# In[139]:


print(Two_Dim[0].shape)


# In[140]:


print(Two_Dim[1].shape)


# In[141]:


Two_Dim_Flat = np.hstack((np.reshape(Two_Dim[0], (-1, 1)), np.reshape(Two_Dim[1], (-1, 1))))


# In[142]:


fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = [15,5])


# In[143]:


#O(n^2)
for i in range(len(spread)):
    parzen_1d = ParzenWindow(df[['petal_length', 'species']], One_Dim, spread[i])
    axs[i].set_prop_cycle(None)
    
    for species in np.unique(df.species):
        axs[i].plot(df['petal_length'][df.species == species], np.zeros_like(df['petal_length'][df.species == species]), 'o', markerfacecolor = 'none')
        
    axs[i].set_prop_cycle(None)
    
    for species in np.unique(df.species):
        axs[i].plot(np.mean(df['petal_length'][df.species == species]), 0, 'x', markersize = 15)
        
    axs[i].set_prop_cycle(None)
    ordered_indices = np.argsort(One_Dim, axis = 0)
    
    for j in range(len(parzen_1d[:,0])):
        axs[i].plot(np.sort(One_Dim), parzen_1d[j, ordered_indices].flatten())
        
    axs[i].set_xlabel('petal_length')
    axs[i].set_title('Gaussian Kernel with h = ' + str(spread[i]))
    
plt.tight_layout()

plt.savefig('parzen_1d.png')

with Image.open('parzen_1d.png') as en:
    en.show()


# In[144]:


fig, axs = plt.subplots(nrows = 1, ncols = 3, figsize = [15,5])


# In[195]:


#O(n^2)
for i in range(len(spread)):
    parzen_2d = ParzenWindow(df[['petal_width', 'petal_length', 'species']], Two_Dim_Flat, spread[i])
    axs[i].set_prop_cycle(None)
    
    for species in np.unique(df.species):
        axs[i].plot(np.mean(df['petal_width'][df.species == species]), np.mean(df['petal_length'][df.species == species]), 'x', markersize = 15)
        
    #axs[i].set_proper_cycle(None)
    mask = parzen_2d<1e-9
    masked = np.copy(parzen_2d)
    masked[mask] = np.nan
    
    for j in range(len(parzen_2d[:, 0])):
        axs[i].contour(Two_Dim[0], Two_Dim[1], np.reshape(masked[j, :], np.shape(Two_Dim[0])), colors = 'C' + str(j))
        
    pred = np.argmax(parzen_2d, axis = 0)
    
    axs[i].contour(Two_Dim[0], Two_Dim[1], np.reshape(pred, np.shape(Two_Dim[0])), colors = 'k')
    
    axs[i].set_xlabel('petal_width')
    axs[i].set_ylabel('petal_length')
    axs[i].set_title('Gaussian Kernel with h = ' + str(spread[i]))
    
plt.tight_layout()

plt.savefig('parzen_2d.png')

with Image.open('parzen_2d.png') as en:
    en.show()

