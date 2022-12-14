#!/usr/bin/env python
# coding: utf-8

# In[91]:


#Roger H Hayden III
#Algorithms for DS
#Programming Assignement 1


# In[1]:


#Importing packages
import pandas as pd
import numpy as np
import statistics 
from scipy import stats
import matplotlib.pylab as plt
import matplotlib.cm as cm
from scipy.fftpack import dct, idct
from sklearn.decomposition import PCA
from tabulate import tabulate
import plotly.express as px
from scipy.stats import chi2
import seaborn as sns


# ----------------------------------PROBLEM 1------------------------------------

# In[2]:


#Read in Data
df = pd.read_csv(r'C:\Users\roger\OneDrive\Desktop\iris.csv')
print(df)

Seperating out Species and Looking at their data
# In[3]:


#Setosa is the first 50 rows and keep all columns
setosa_species = df.iloc[0:50,:]
print(setosa_species)


# In[4]:


#Versicolor is the second 50 rows and keep all columns
versicolor_species = df.iloc[51:100,:]
print(versicolor_species)


# In[5]:


#Virginica is the third 50 rows and keep all columns
virginica_species = df.iloc[100:150,:]
print(virginica_species)


# Filtering by Each Attribute

# sepal_length does not keep the species in the same order 
# sepal_length does not keep the species grouped together 

# In[6]:


#Setting Pandas to allow us to display all of the rows
pd.set_option('display.max_rows', 500)
df.sort_values(by = 'sepal_length', ascending = False)


# sepal_width does not keep the species in the same order 
# sepal_width does not keep the species grouped together 

# In[7]:


df.sort_values(by = 'sepal_width', ascending = False)


# sepal_length does not keep the species in the same order 
# sepal_length does not keep the species grouped together 
# 
# This is very close to doing so, but some virginica and versicolor overlap

# In[8]:


df.sort_values(by = 'petal_length', ascending = False)


# sepal_width does not keep the species in the same order 
# sepal_width does not keep the species grouped together 
# 
# Some virginica and versicolor overlap

# In[9]:


df.sort_values(by = 'petal_width', ascending = False)


# -------------------------------PROBLEM 2---------------------------------------

# Setosa Stats

# In[10]:


#Minimum Values
#minimum setosa sepal length
minssl = min(setosa_species.loc[:,'sepal_length'])

#minimum setosa sepal width
minssw = min(setosa_species.loc[:,'sepal_width'])

#minimum setosa petal length
minspl = min(setosa_species.loc[:,'petal_length'])

#minimum setosa petal width
minspw = min(setosa_species.loc[:,'petal_width'])


# In[11]:


#Maximum Values
#maximum setosa sepal length
maxssl = max(setosa_species.loc[:,'sepal_length'])

#maximum setosa sepal width
maxssw = max(setosa_species.loc[:,'sepal_width'])

#maximum setosa petal length
maxspl = max(setosa_species.loc[:,'petal_length'])

#maximum setosa petal width
maxspw = max(setosa_species.loc[:,'petal_width'])


# In[12]:


#Mean
#mean setosa sepal length
meanssl = statistics.mean(setosa_species.loc[:,'sepal_length'])

#mean setosa sepal width
meanssw = statistics.mean(setosa_species.loc[:,'sepal_width'])

#mean setosa petal length
meanspl = statistics.mean(setosa_species.loc[:,'petal_length'])

#mean setosa petal width
meanspw = statistics.mean(setosa_species.loc[:,'petal_width'])


# In[13]:


#Trimmed Mean
#trimmed mean setosa sepal length
tmssl = stats.trim_mean(setosa_species.loc[:,'sepal_length'], 
                         proportiontocut=0.2)

#trimmed mean setosa sepal width
tmssw = stats.trim_mean(setosa_species.loc[:,'sepal_width'], 
                      proportiontocut=0.2)

#trimmed mean setosa petal length
tmspl = stats.trim_mean(setosa_species.loc[:,'petal_length'], 
                      proportiontocut=0.2)

#trimmed mean setosa petal width
tmspw = stats.trim_mean(setosa_species.loc[:,'petal_width'], 
                      proportiontocut=0.2)


# In[14]:


#Standard Deviation
#standard deviation setosa sepal length
stdssl = statistics.stdev(setosa_species.loc[:,'sepal_length'])

#standard deviation setosa sepal width 
stdssw = statistics.stdev(setosa_species.loc[:,'sepal_width'])

#standard deviation setosa petal length
stdspl = statistics.stdev(setosa_species.loc[:,'petal_length'])

#standard deviation petal width
stdspw = statistics.stdev(setosa_species.loc[:,'petal_width'])


# In[15]:


#Skewness
#skewness setosa sepal length
skewssl = stats.skew(setosa_species.loc[:,'sepal_length'])

#skewness setosa sepal width
skewssw = stats.skew(setosa_species.loc[:,'sepal_width'])

#skewness setosa petal length
skewspl = stats.skew(setosa_species.loc[:,'petal_length'])

#skewness setosa petal width
skewspw = stats.skew(setosa_species.loc[:,'petal_width'])


# In[16]:


#Kurtosis
#kurtosis setosa sepal length
kurtssl = stats.kurtosis(setosa_species.loc[:,'sepal_length'])

#kurtosis setosa sepal width
kurtssw = stats.kurtosis(setosa_species.loc[:,'sepal_width'])

#kurtosis setosa petal length
kurtspl = stats.kurtosis(setosa_species.loc[:,'petal_length'])

#kurtosis petal width
kurtspw = stats.kurtosis(setosa_species.loc[:,'petal_width'])


# Versicolor Stats

# In[17]:


#Minimum Values
#minimum Versicolor sepal length
minvsl = min(versicolor_species.loc[:,'sepal_length'])

#minimum Versicolor sepal width
minvsw = min(versicolor_species.loc[:,'sepal_width'])

#minimum Versicolor petal length
minvpl = min(versicolor_species.loc[:,'petal_length'])

#minimum Versicolor petal width
minvpw = min(versicolor_species.loc[:,'petal_width'])


# In[18]:


#Maximum Values
#maximum Versicolor sepal length
maxvsl = max(versicolor_species.loc[:,'sepal_length'])

#maximum Versicolor sepal width
maxvsw = max(versicolor_species.loc[:,'sepal_width'])

#maximum Versicolor petal length
maxvpl = max(versicolor_species.loc[:,'petal_length'])

#maximum Versicolor petal width
maxvpw = max(versicolor_species.loc[:,'petal_width'])


# In[19]:


#Mean
#mean setosa sepal length
meanvsl = statistics.mean(versicolor_species.loc[:,'sepal_length'])

#mean setosa sepal width
meanvsw = statistics.mean(versicolor_species.loc[:,'sepal_width'])

#mean setosa petal length
meanvpl = statistics.mean(versicolor_species.loc[:,'petal_length'])

#mean setosa petal width
meanvpw = statistics.mean(versicolor_species.loc[:,'petal_width'])


# In[20]:


#Trimmed Mean
#trimmed mean setosa sepal length
tmvsl = stats.trim_mean(versicolor_species.loc[:,'sepal_length'], 
                         proportiontocut=0.2)

#trimmed mean setosa sepal width
tmvsw = stats.trim_mean(versicolor_species.loc[:,'sepal_width'], 
                      proportiontocut=0.2)

#trimmed mean setosa petal length
tmvpl = stats.trim_mean(versicolor_species.loc[:,'petal_length'], 
                      proportiontocut=0.2)

#trimmed mean setosa petal width
tmvpw = stats.trim_mean(versicolor_species.loc[:,'petal_width'], 
                      proportiontocut=0.2)


# In[21]:


#Standard Deviation
#standard deviation setosa sepal length
stdvsl = statistics.stdev(versicolor_species.loc[:,'sepal_length'])

#standard deviation setosa sepal width 
stdvsw = statistics.stdev(versicolor_species.loc[:,'sepal_width'])

#standard deviation setosa petal length
stdvpl = statistics.stdev(versicolor_species.loc[:,'petal_length'])

#standard deviation petal width
stdvpw = statistics.stdev(versicolor_species.loc[:,'petal_width'])


# In[22]:


#Skewness
#skewness setosa sepal length
skewvsl = stats.skew(versicolor_species.loc[:,'sepal_length'])

#skewness setosa sepal width
skewvsw = stats.skew(versicolor_species.loc[:,'sepal_width'])

#skewness setosa petal length
skewvpl = stats.skew(versicolor_species.loc[:,'petal_length'])

#skewness setosa petal width
skewvpw = stats.skew(versicolor_species.loc[:,'petal_width'])


# In[23]:


#Kurtosis
#kurtosis setosa sepal length
kurtvsl = stats.kurtosis(versicolor_species.loc[:,'sepal_length'])

#kurtosis setosa sepal width
kurtvsw = stats.kurtosis(versicolor_species.loc[:,'sepal_width'])

#kurtosis setosa petal length
kurtvpl = stats.kurtosis(versicolor_species.loc[:,'petal_length'])

#kurtosis petal width
kurtvpw = stats.kurtosis(versicolor_species.loc[:,'petal_width'])


# Virginica Stats

# In[24]:


#Minimum Values
#minimum Virginica sepal length
minvisl = min(virginica_species.loc[:,'sepal_length'])

#minimum Virginica sepal width
minvisw = min(virginica_species.loc[:,'sepal_width'])

#minimum Virginica petal length
minvipl = min(virginica_species.loc[:,'petal_length'])

#minimum Virginica petal width
minvipw = min(virginica_species.loc[:,'petal_width'])


# In[25]:


#Maximum Values
#maximum Virginica sepal length
maxvisl = max(virginica_species.loc[:,'sepal_length'])

#maximum Virginica sepal width
maxvisw = max(virginica_species.loc[:,'sepal_width'])

#maximum Virginica petal length
maxvipl = max(virginica_species.loc[:,'petal_length'])

#maximum Virginica petal width
maxvipw = max(virginica_species.loc[:,'petal_width'])


# In[26]:


#Mean
#mean setosa sepal length
meanvisl = statistics.mean(virginica_species.loc[:,'sepal_length'])

#mean setosa sepal width
meanvisw = statistics.mean(virginica_species.loc[:,'sepal_width'])

#mean setosa petal length
meanvipl = statistics.mean(virginica_species.loc[:,'petal_length'])

#mean setosa petal width
meanvipw = statistics.mean(virginica_species.loc[:,'petal_width'])


# In[27]:


#Trimmed Mean
#trimmed mean setosa sepal length
tmvisl = stats.trim_mean(virginica_species.loc[:,'sepal_length'], 
                         proportiontocut=0.2)

#trimmed mean setosa sepal width
tmvisw = stats.trim_mean(virginica_species.loc[:,'sepal_width'], 
                      proportiontocut=0.2)

#trimmed mean setosa petal length
tmvipl = stats.trim_mean(virginica_species.loc[:,'petal_length'], 
                      proportiontocut=0.2)

#trimmed mean setosa petal width
tmvipw = stats.trim_mean(virginica_species.loc[:,'petal_width'], 
                      proportiontocut=0.2)


# In[28]:


#Standard Deviation
#standard deviation setosa sepal length
stdvisl = statistics.stdev(virginica_species.loc[:,'sepal_length'])

#standard deviation setosa sepal width 
stdvisw = statistics.stdev(virginica_species.loc[:,'sepal_width'])

#standard deviation setosa petal length
stdvipl = statistics.stdev(virginica_species.loc[:,'petal_length'])

#standard deviation petal width
stdvipw = statistics.stdev(virginica_species.loc[:,'petal_width'])


# In[29]:


#Skewness
#skewness setosa sepal length
skewvisl = stats.skew(virginica_species.loc[:,'sepal_length'])

#skewness setosa sepal width
skewvisw = stats.skew(virginica_species.loc[:,'sepal_width'])

#skewness setosa petal length
skewvipl = stats.skew(virginica_species.loc[:,'petal_length'])

#skewness setosa petal width
skewvipw = stats.skew(virginica_species.loc[:,'petal_width'])


# In[30]:


#Kurtosis
#kurtosis setosa sepal length
kurtvisl = stats.kurtosis(virginica_species.loc[:,'sepal_length'])

#kurtosis setosa sepal width
kurtvisw = stats.kurtosis(virginica_species.loc[:,'sepal_width'])

#kurtosis setosa petal length
kurtvipl = stats.kurtosis(virginica_species.loc[:,'petal_length'])

#kurtosis petal width
kurtvipw = stats.kurtosis(virginica_species.loc[:,'petal_width'])


# Statistical Table

# In[31]:


table = [['Statistic', 'Species', 'Sepal Len.','Sepal Wid.', 'Petal Len.', 'Petal Wid.'], 
        ['Minimum','Setosa', minssl, minssw, minspl, minspw],
        ['', 'Versicolor', minvsl, minvsw, minvpl, minvpw],
        ['', 'Virginica', minvisl, minvisw, minvipl, minvipw],
        ['Maximum', 'Setosa', maxssl, maxssw, maxspl, maxspw],
        ['', 'Versicolor', maxvsl, maxvsw, maxvpl, maxvpw],
        ['', 'Virginica', maxvisl, maxvisw, maxvipl, maxvipw],
        ['Mean', 'Setosa', meanssl, meanssw, meanspl, meanspw],
        ['', 'Versicolor', meanvsl, meanvsw, meanvpl, meanvpw],
        ['', 'Virginica', meanvisl, meanvisw, meanvipl, meanvipw],
        ['Trimmed Mean', 'Setosa',tmssl, tmssw, tmspl, tmspw],
        ['', 'Versicolor', tmvsl, tmvsw, tmvpl, tmvpw],
        ['', 'Virginica', tmvisl, tmvisw, tmvipl, tmvipw],
        ['Std. Dev.', 'Setosa', stdssl, stdssw, stdspl, stdspw],
        ['', 'Versicolor', stdvsl, stdvsw, stdvpl, stdvpw],
        ['', 'Virginica', stdvisl, stdvisw, stdvipl, stdvipw],
        ['Skewness', 'Setosa', skewssl, skewssw, skewspl, skewspw],
        ['', 'Versicolor', skewvsl, skewvsw, skewvpl, skewvpw],
        ['', 'Virginica', skewvisl, skewvisw, skewvipl, skewvipw],
        ['Kurtosis', 'Setosa', kurtssl, kurtssw, kurtspl, kurtspw],
        ['', 'Versicolor', kurtvsl, kurtvsw, kurtvpl, kurtvpw],
        ['', 'Virginica', kurtvisl, kurtvisw, kurtvipl, kurtvipw]]
print(tabulate(table, headers = 'firstrow'))


# ------------------------PROBLEM 3------------------------

# PART A

# Plotting sepal length vs. sepal width with regards to species

# In[32]:


fig = px.scatter(df, x="sepal_length",
                 y="sepal_width", 
                 color="species")
  
fig.show()


# PART B

# Outlier Removal with Mahalanobis Distance

# In[33]:


df2 = pd.DataFrame(df,columns=['sepal_length', 'sepal_width',
                                'petal_length','petal_width'])


# In[34]:


def calculateMahalanobis(y=None, data=None, cov=None):
  
    y_mu = y - np.mean(data)
    if not cov:
        cov = np.cov(data.values.T)
    inv_covmat = np.linalg.inv(cov)
    left = np.dot(y_mu, inv_covmat)
    mahal = np.dot(left, y_mu.T)
    return mahal.diagonal()


# In[35]:


df['Mahalanobis'] = calculateMahalanobis(y=df2, data=df2[[
  'sepal_length', 'sepal_width', 'petal_length','petal_width']])


# In[36]:


print(df)


# In[37]:


df['p'] = 1 - chi2.cdf(df['Mahalanobis'], 3)


# In[38]:


print(df)


# In[39]:


#Consider Outliers p values < 0.01
Outliers = []
Outliers = df.loc[df.p < 0.01].index.values.tolist()
print(Outliers)


# In[40]:


#Iris Outlier Removed
iris_OR = df[~df.index.isin(Outliers)]
iris_OR.shape


# In[41]:


print(iris_OR)


# In[42]:


#Box Plot for sepal width and sepal length
sns.boxplot(x="sepal_width", y="sepal_length", data=iris_OR)


# In[43]:


#Boxplot for all 4 features and all species
sns.boxplot(data = iris_OR.iloc[:,0:4])


# In[44]:


#Boxplot for all 4 features and the setosa species
sns.boxplot(data = iris_OR.iloc[0:49,0:4])


# In[45]:


#Boxplot for all 4 features and the Versicolor species
sns.boxplot(data = iris_OR.iloc[50:99,0:4])


# In[46]:


#Boxplot for all 4 features and the Virginica species
sns.boxplot(data = iris_OR.iloc[100:149,0:4])


# It appears there still may potentially be some outliers - specifically:
# 
# Setosa:
#     - Appears to be some petal length value(s) just below the box plot 
#     - Appears to be some petal width value(s) just above
# 
# Versicolor:
#     - Appears to be some petal length value(s) just below the box plot
#     
# Virginica:
#     - Appears to be some sepal length value(s) just below the box plot
#     - Appears to be some sepal width value(s) above AND below the box plot
#     - Appears to have some petal length value(s) just above the box plot
#     
# All of these outliers were determined by me reviewing each box plot to see where some points may lie outside the range of the boxplot. There are other ways of determining outliers as well but this is the one that was easiest for me as I struggled getting the graph with the ovals. 
# 
# In this instance I used the Mahalanobis Distance and P - value to determine where outliers may be as this is what we discussed in office hours about the recommended way to do this problem.

# PART C

# Mean Values

# In[55]:


#Setosa
Setosa_Mean = []
Setosa_Mean = [meanssl, meanssw, meanspl,meanspw]
print("The Setosa Mean Values are: ") 
Setosa_Mean


# In[56]:


#Versicolor
Versi_Mean = []
Versi_Mean = [meanvsl, meanvsw, meanvpl, meanvpw]
print("The Versicolor Mean Values are: ") 
Versi_Mean


# In[57]:


#Virginica
Vir_Mean = []
Vir_Mean = [meanvisl, meanvisw, meanvipl, meanvipw]
print("The Virginica Mean Values are: ") 
Vir_Mean


# Variance Values

# In[58]:


#Setosa
Setosa_Var = []
Setosa_Var = [statistics.variance(setosa_species.loc[:,'sepal_length']),
             statistics.variance(setosa_species.loc[:,'sepal_width']),
             statistics.variance(setosa_species.loc[:,'petal_length']),
             statistics.variance(setosa_species.loc[:,'petal_width'])]
print("The Setosa Variance Values are: ") 
Setosa_Var


# In[59]:


#Versicolor
Versi_Var = []
Versi_Var = [statistics.variance(versicolor_species.loc[:,'sepal_length']),
             statistics.variance(versicolor_species.loc[:,'sepal_width']),
             statistics.variance(versicolor_species.loc[:,'petal_length']),
             statistics.variance(versicolor_species.loc[:,'petal_width'])]
print("The Versicolor Variance Values are: ") 
Versi_Var


# In[60]:


#Virginica
Vir_Var = []
Vir_Var = [statistics.variance(virginica_species.loc[:,'sepal_length']),
             statistics.variance(virginica_species.loc[:,'sepal_width']),
             statistics.variance(virginica_species.loc[:,'petal_length']),
             statistics.variance(virginica_species.loc[:,'petal_width'])]
print("The Virginica Variance Values are: ") 
Vir_Var


# Feature Rankings

# In[78]:


Setosa_FR = []
Versicolor_FR = []
Virginica_FR = []

def Feat_Rank(mean, var):
    feature_rank = []
    sum = 0
    for i in range(0,4):
        for j in range(0,4):
            if i != j:
                sum+=pow(mean[i]-mean[j], 2)/(var[i]+var[j])
            feature_rank.append(sum)
            sum=0     
    return feature_rank

Setosa_FR = Feat_Rank(Setosa_Mean, Setosa_Var)
Versicolor_FR = Feat_Rank(Versi_Mean, Versi_Var)
Virginica_FR = Feat_Rank(Vir_Mean, Vir_Var)


# In[79]:


print("The Setosa Feature Rankings are: ")
Setosa_FR


# In[80]:


print("The Versicolor Feature Rankings are: ")
Versicolor_FR


# In[81]:


print("The Virginica Feature Rankings are: ")
Virginica_FR


# In[82]:


def Multi_Rank(x, y, z):
    sum = 0
    Feature_Sum = []
    for i in range(0,4):
        sum+=x[i]+y[i]+z[i]
        Feature_Sum.append(sum)
    return Feature_Sum

Multi_Rank(Setosa_FR, Versicolor_FR, Virginica_FR)


# ------------------------PROBLEM 4------------------------

# In[89]:


"""
Created on Thu Sep 26 08:45:20 2019

@author: Samuel Fisher, Intern
Johns Hopkins University Applied Physics Laboratory
"""

#Display who won and add to win counter
def whoWin(x,End,Xwin,Owin): 
    Xwin = 0
    Owin = 0
    if x == 1:
        End.configure(text="Player 1 has won!", background = 'white')
        Xwin = 1
    elif x == 2:
        End.configure(text="Player 2 has won!", background = 'white')
        Owin = 1
    else:
        End.configure(text="Nobody Wins", background = 'white')
    gameover = 1
    L = [Xwin,Owin,gameover]
    return L

#Check if there is a three in a row
#If there is a win, a display which team one and count that win
def checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill): 
    if place[1] == place[0] and place[0] == place[2] and place[1] != 0:
        print ("Player",place[1]," wins")
        return whoWin(place[1],End,Xwin,Owin)
    if place[0] == place[3] and place[0] == place[6] and place[0] != 0:
        print ("Player",place[0]," wins")
        return whoWin(place[0],End,Xwin,Owin)
    if place[0] == place[4] and place[0] == place[8] and place[0] != 0:
        print ("Player",place[0]," wins")
        return whoWin(place[0],End,Xwin,Owin)
    if place[1] == place[4] and place[1] == place[7] and place[1] != 0:
        print ("Player",place[1]," wins")
        return whoWin(place[1],End,Xwin,Owin)
    if place[2] == place[4] and place[2] == place[6] and place[2] != 0:
        print ("Player",place[2]," wins")
        return whoWin(place[2],End,Xwin,Owin)
    if place[2] == place[5] and place[2] == place[8] and place[2] != 0:
        print ("Player",place[2]," wins")
        return whoWin(place[2],End,Xwin,Owin)
    if place[3] == place[4] and place[3] == place[5] and place[3] != 0:
        print ("Player",place[3]," wins")
        return whoWin(place[3],End,Xwin,Owin)
    if place[6] == place[7] and place[8] == place[6] and place[6] != 0:
        print ("Player",place[6]," wins")
        return whoWin(place[7],End,Xwin,Owin)
    tie = 1
    for i in place:
        if i == 0:
            tie = 0
    if tie == 1:
        return whoWin(3,End,Xwin,Owin)
        
    return [0,0,0]

#Check who won without calling whoWin
#Necessary for MiniMax
def checkWin2(place):
    if place[1] == place[0] and place[0] == place[2] and place[1] != 0:
        return place[1]
    if place[0] == place[3] and place[0] == place[6] and place[0] != 0:
        return place[0]
    if place[0] == place[4] and place[0] == place[8] and place[0] != 0:
        return place[0]
    if place[1] == place[4] and place[1] == place[7] and place[1] != 0:
        return place[1]
    if place[2] == place[4] and place[2] == place[6] and place[2] != 0:
        return place[2]
    if place[2] == place[5] and place[2] == place[8] and place[2] != 0:
        return place[2]
    if place[3] == place[4] and place[3] == place[5] and place[3] != 0:
        return place[3]
    if place[6] == place[7] and place[8] == place[6] and place[6] != 0:
        return place[6]
    tie = 1
    for i in place:
        if i == 0:
            tie = 0
    if tie == 1:
        return 0
        
    return [0,0,0]

#Check possibilities for wins in the next move
def checkWinPos(place):
    #Columns
    if abs(place[0] + place[1] + place[2]) == 2:
        if abs(place[0]) != 1:
            return 0
        elif abs(place[1]) != 1:
            return 1
        else:
            return 2 
    if abs(place[3] + place[4] + place[5]) == 2:
        if abs(place[3]) != 1:
            return 3
        elif abs(place[4]) != 1:
            return 4
        else:
            return 5  
    if abs(place[6] + place[7] + place[8]) == 2:
        if abs(place[6]) != 1:
            return 6
        elif abs(place[7]) != 1:
            return 7
        else:
            return 8 
    #Rows
    if abs(place[0] + place[3] + place[6]) == 2:
        if abs(place[0]) != 1:
            return 0
        elif abs(place[3]) != 1:
            return 3
        else:
            return 6
    if abs(place[1] + place[4] + place[7]) == 2:
        if abs(place[1]) != 1:
            return 1
        elif abs(place[4]) != 1:
            return 4
        else:
            return 7
    if abs(place[2] + place[5] + place[8]) == 2:
        if abs(place[2]) != 1:
            return 2
        elif abs(place[5]) != 1:
            return 5
        else:
            return 8
    #Diagonal
    if abs(place[0] + place[4] + place[8]) == 2:
        if abs(place[0]) != 1:
            return 0
        elif abs(place[4]) != 1:
            return 4
        else:
            return 8
    if abs(place[2] + place[4] + place[6]) == 2:
        if abs(place[2]) != 1:
            return 2
        elif abs(place[4]) != 1:
            return 4
        else:
            return 6
        
    


# In[90]:


"""
Created on Wed Sep 18 15:56:53 2019

@author: Samuel Fisher, Intern
Johns Hopkins University Applied Physics Laboratory
"""
import tkinter
import sys
#import cv2
#import PIL.Image, PIL.ImageTk
import random
sys.setrecursionlimit(2000)#Add limit for recursion
from checkWin_Incomplete import checkWin
#from checkWin_Incomplete import checkWin2
from checkWin_Incomplete import checkWinPos
game = tkinter.Toplevel()#init board
game.geometry("350x400+300+300") #set base dimensions


boardCanvas = tkinter.Canvas(game, width = 640, height = 640)#Initialize TKinter canvas

AIturn = 0 #AI goes second
#boardImage = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(board))  #Load Board
boardCanvas.create_image(50, 89,anchor= tkinter.NW)
boardCanvas.create_line(114, 90, 114, 280)
boardCanvas.create_line(173, 90, 173, 280)
boardCanvas.create_line(50, 153, 240, 153)
boardCanvas.create_line(50, 215, 240, 215)
aiSkill = 1 #Change to 0 for no AI, Change to 1 for Easy, Change to 2 for Medium, 3 for Hard
# Easy follows optimal move pattern if no places are marked
# Medium is easy with additional conditional statements
# Hard uses the MiniMax algorithm
aiX = 0
def decisionMaker(boardState,minimax,depth):
    while True: #Delete this loop if attempting MiniMax
        c = random.randint(0,8)
        if boardState[c] == 0:
            return c
    #Do MiniMax Here
    
def getPlace():
    global place
    return place


def AI(aiSkill,place,turn,AIturn):
    AIturn = AIturn+1 #Alternate player and AI
    if aiSkill == 1: #Easy
        if place[4] == 0:
            midMidPress()
        elif place[0] == 0:
            topLeftPress()
        elif place[2] == 0:
            botLeftPress()
        elif place[6] == 0:
            topRightPress()
        elif place[8] == 0:
            botRightPress()
        elif place[1] == 0:
            midLeftPress()
        elif place[3] == 0:
            topMidPress()
        elif place[5] == 0:
            botMidPress()
        elif place[7] == 0:
            midRightPress()
    if aiSkill == 2: #Medium
        F = checkWinPos(place)
        if F != None:
            if F == 0:
                topLeftPress()
            if F == 1:
                midLeftPress()
            if F == 2:
                botLeftPress()
            if F == 3:
                topMidPress()
            if F == 4:
                midMidPress()
            if F == 5:
                botMidPress()
            if F == 6:
                topRightPress()
            if F == 7:
                midRightPress()
            if F == 8:
                botRightPress()
        elif place[4] == 0:
            midMidPress()
        elif place[0] == 0:
            topLeftPress()
        elif place[2] == 0:
            botLeftPress()
        elif place[6] == 0:
            topRightPress()
        elif place[8] == 0:
            botRightPress()
        elif place[1] == 0:
            midLeftPress()
        elif place[3] == 0:
            topMidPress()
        elif place[5] == 0:
            botMidPress()
        elif place[7] == 0:
            midRightPress()
    if aiSkill == 3: #Hard
        G = [] #Create new list G. If G = place, python thinks it is actually place under a different name.
        for i in place:
            G.append(i)

        F = decisionMaker(G,0,0)
        if F == 0:
            topLeftPress()
        if F == 1:
            midLeftPress()
        if F == 2:
            botLeftPress()
        if F == 3:
            topMidPress()
        if F == 4:
            midMidPress()
        if F == 5:
            botMidPress()
        if F == 6:
            topRightPress()
        if F == 7:
            midRightPress()
        if F == 8:
            botRightPress()
            
#initialize filled places and whose turn it is
place = [0,0,0,0,0,0,0,0,0]
turn = 0

#Initialize variables to describe whether boxes are marked or not
topLeftComp = 0
midLeftComp = 0
botLeftComp = 0
topMidComp = 0
midMidComp = 0
botMidComp = 0
topRightComp = 0
midRightComp = 0
botRightComp = 0

#Initialize variables for after the game is completed. 
#Counts wins by each player and stops moves when game is over.
gameOver=0
Xwin = 0
Owin = 0

L = []
#button press functions
def topLeftPress():
    global gameOver #No moves can be made if game is over.
    if gameOver == 0:
        global turn #These are required to call the function in the checkWin.py
        global topLeftComp
        global AIturn
        global Xwin
        global Owin
        global aiSkill
        if topLeftComp == 0: #check if space is filled   
            if turn == 0:
                place[0] = 1
                turn = 1
                TopLeft.configure(text=("X"))
            else:
                turn = 0
                place[0] = -1
                TopLeft.configure(text="O")
            topLeftComp = 1
            L = checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill)
            Xwin = Xwin+L[0]
            Owin = Owin+L[1]
            gameOver = L[2]
            if AIturn%2 == 0: #Call AI turn every two turns
                AIturn = AIturn+1
                AI(aiSkill,place,turn,AIturn)
                
            else:
                AIturn = AIturn+1
        else:
            print("Already Set Box")
        return place
def midLeftPress():
    global gameOver
    if gameOver == 0:
        global turn
        global midLeftComp
        global AIturn
        global Xwin
        global Owin
        global aiSkill
        if midLeftComp == 0:    
            if turn == 0:
                place[1] = 1
                turn = 1
                MidLeft.configure(text="X")
            else:
                turn = 0
                place[1] = -1
                MidLeft.configure(text="O")
            midLeftComp = 1
            L = checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill)
            
            Xwin = Xwin+L[0]
            Owin = Owin+L[1]
            gameOver = L[2]
            if AIturn%2 == 0:
                AIturn = AIturn+1
                AI(aiSkill,place,turn,AIturn)
                
            else:
                AIturn = AIturn+1
        else:
            print("Already Set Box")
        return place
def botLeftPress():
    global gameOver
    if gameOver == 0:
        global turn
        global botLeftComp
        global AIturn
        global Xwin
        global Owin
        global aiSkill
        if botLeftComp == 0:    
            if turn == 0:
                place[2] = 1
                turn = 1
                BotLeft.configure(text="X")
            else:
                turn = 0
                place[2] = -1
                BotLeft.configure(text="O")
            botLeftComp = 1
            L = checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill)
            Xwin = Xwin+L[0]
            Owin = Owin+L[1]
            gameOver = L[2]
            if AIturn%2 == 0:
                AIturn = AIturn+1
                AI(aiSkill,place,turn,AIturn)
                
            else:
                AIturn = AIturn+1
        else:
            print("Already Set Box")
        return place
def topMidPress():
    global gameOver
    if gameOver == 0:
        global turn
        global topMidComp
        global AIturn
        global Xwin
        global Owin
        global aiSkill
        if topMidComp == 0:    
            if turn == 0:
                place[3] = 1
                turn = 1
                TopMid.configure(text="X")
            else:
                turn = 0
                place[3] = -1
                TopMid.configure(text="O")
            topMidComp = 1
            L = checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill)
            Xwin = Xwin+L[0]
            Owin = Owin+L[1]
            gameOver = L[2]
            if AIturn%2 == 0:
                AIturn = AIturn+1
                AI(aiSkill,place,turn,AIturn)
                
            else:
                AIturn = AIturn+1
        else:
            print("Already Set Box")
        return place
def midMidPress():
    global gameOver
    if gameOver == 0:
        global turn
        global midMidComp
        global AIturn
        global Xwin
        global Owin
        global aiSkill
        if midMidComp == 0:    
            if turn == 0:
                place[4] = 1
                turn = 1
                MidMid.configure(text="X")
            else:
                turn = 0
                place[4] = -1
                MidMid.configure(text="O")
            midMidComp = 1
            L = checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill)
            Xwin = Xwin+L[0]
            Owin = Owin+L[1]
            gameOver = L[2]
            if AIturn%2 == 0:
                AIturn = AIturn+1
                AI(aiSkill,place,turn,AIturn)
                
            else:
                AIturn = AIturn+1
        else:
            print("Already Set Box")
        return place
def botMidPress():
    global gameOver
    if gameOver == 0:
        global turn
        global botMidComp
        global AIturn
        global Xwin
        global Owin
        global aiSkill
        if botMidComp == 0:    
            if turn == 0:
                place[5] = 1
                turn = 1
                BotMid.configure(text="X")
            else:
                turn = 0
                place[5] = -1
                BotMid.configure(text="O")
            botMidComp = 1
            L = checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill)
            Xwin = Xwin+L[0]
            Owin = Owin+L[1]
            gameOver = L[2]
            if AIturn%2 == 0:
                AIturn = AIturn+1
                AI(aiSkill,place,turn,AIturn)
                
            else:
                AIturn = AIturn+1
        else:
            print("Already Set Box")
        return place
def topRightPress():
    global gameOver
    if gameOver == 0:
        global turn
        global topRightComp
        global AIturn
        global Xwin
        global Owin
        global aiSkill
        if topRightComp == 0:    
            if turn == 0:
                place[6] = 1
                turn = 1
                TopRight.configure(text="X")
            else:
                turn = 0
                place[6] = -1
                TopRight.configure(text="O")
            topRightComp = 1
            L = checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill)
            Xwin = Xwin+L[0]
            Owin = Owin+L[1]
            gameOver = L[2]
            if AIturn%2 == 0:
                AIturn = AIturn+1
                AI(aiSkill,place,turn,AIturn)
                
            else:
                AIturn = AIturn+1
        else:
            print("Already Set Box")
        return place
def midRightPress():
    global gameOver
    if gameOver == 0:
        global turn
        global midRightComp
        global AIturn
        global Xwin
        global Owin
        global aiSkill
        if midRightComp == 0:    
            if turn == 0:
                place[7] = 1
                turn = 1
                MidRight.configure(text="X")
            else:
                turn = 0
                place[7] = -1
                MidRight.configure(text="O")
            midRightComp = 1
            L = checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill)
            Xwin = Xwin+L[0]
            Owin = Owin+L[1]
            gameOver = L[2]
            if AIturn%2 == 0:
                AIturn = AIturn+1
                AI(aiSkill,place,turn,AIturn)
                
            else:
                AIturn = AIturn+1
        else:
            print("Already Set Box")
        return place
def botRightPress():
    global gameOver
    if gameOver == 0:
        global turn
        global botRightComp
        global AIturn
        global Xwin
        global Owin
        global aiSkill
        if botRightComp == 0:    
            if turn == 0:
                place[8] = 1
                turn = 1
                BotRight.configure(text="X")
            else:
                turn = 0
                place[8] = -1
                BotRight.configure(text="O")
            botRightComp = 1
            L = checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill)
            Xwin = Xwin+L[0]
            Owin = Owin+L[1]
            gameOver = L[2]
            if AIturn%2 == 0:
                AIturn = AIturn+1
                AI(aiSkill,place,turn,AIturn)
                
            else:
                AIturn = AIturn+1
        else:
            print("Already Set Box")
        return place

#reset all board variables
def Reset():
    global place
    global turn
    global topLeftComp
    global midLeftComp
    global botLeftComp
    global topMidComp
    global midMidComp
    global botMidComp
    global topRightComp
    global midRightComp
    global botRightComp
    global gameOver
    global AIturn
    AIturn = 0
    place = [0,0,0,0,0,0,0,0,0]
    turn = 0
    topLeftComp = 0
    midLeftComp = 0
    botLeftComp = 0
    topMidComp = 0
    midMidComp = 0
    botMidComp = 0
    topRightComp = 0
    midRightComp = 0
    botRightComp = 0
    TopLeft.configure(text=" ")
    MidLeft.configure(text=" ")
    BotLeft.configure(text=" ")
    TopMid.configure(text=" ")
    MidMid.configure(text=" ")
    BotMid.configure(text=" ")
    TopRight.configure(text=" ")
    MidRight.configure(text=" ")
    BotRight.configure(text=" ")
    End.configure(width = 18, height = 1, background = "#F0F0F0", activebackground = "white", relief = "flat", font=('courier',14),text = " ")
    gameOver = 0
    Score.configure(text = ("Score", Xwin, ":", Owin))
    

#Button Setup
TopLeft = tkinter.Button(boardCanvas, text = " ", command = topLeftPress)
TopLeft.configure(width = 7, height = 3, background = "#F0F0F0", activebackground = "#F0F0F0", relief = "flat")
TopLeft_window = boardCanvas.create_window(50, 90, anchor=tkinter.NW, window=TopLeft)

MidLeft = tkinter.Button(boardCanvas, text = " ", command = midLeftPress)
MidLeft.configure(width = 7, height = 3, background = "#F0F0F0", activebackground = "#F0F0F0", relief = "flat")
MidLeft_window = boardCanvas.create_window(50, 158, anchor=tkinter.NW, window=MidLeft)

BotLeft = tkinter.Button(boardCanvas, text = " ", command = botLeftPress)
BotLeft.configure(width = 7, height = 3, background = "#F0F0F0", activebackground = "#F0F0F0", relief = "flat")
BotLeft_window = boardCanvas.create_window(50, 220, anchor=tkinter.NW, window=BotLeft)

TopMid = tkinter.Button(boardCanvas, text = " ", command = topMidPress)
TopMid.configure(width = 6, height = 3, background = "#F0F0F0", activebackground = "#F0F0F0", relief = "flat")
TopMid_window = boardCanvas.create_window(118, 90, anchor=tkinter.NW, window=TopMid)

MidMid = tkinter.Button(boardCanvas, text = " ", command = midMidPress)
MidMid.configure(width = 6, height = 3, background = "#F0F0F0", activebackground = "#F0F0F0", relief = "flat")
MidMid_window = boardCanvas.create_window(118, 158, anchor=tkinter.NW, window=MidMid)

BotMid = tkinter.Button(boardCanvas, text = " ", command = botMidPress)
BotMid.configure(width = 6, height = 3, background = "#F0F0F0", activebackground = "#F0F0F0", relief = "flat")
BotMid_window = boardCanvas.create_window(118, 224, anchor=tkinter.NW, window=BotMid)

TopRight = tkinter.Button(boardCanvas, text = " ", command = topRightPress)
TopRight.configure(width = 8, height = 3, background = "#F0F0F0", activebackground = "#F0F0F0", relief = "flat")
TopRight_window = boardCanvas.create_window(180, 90, anchor=tkinter.NW, window=TopRight)

MidRight = tkinter.Button(boardCanvas, text = " ", command = midRightPress)
MidRight.configure(width = 8, height = 3, background = "#F0F0F0", activebackground = "#F0F0F0", relief = "flat")
MidRight_window = boardCanvas.create_window(180, 158, anchor=tkinter.NW, window=MidRight)

BotRight = tkinter.Button(boardCanvas, text = " ", command = botRightPress)
BotRight.configure(width = 8, height = 3, background = "#F0F0F0", activebackground = "#F0F0F0", relief = "flat")
BotRight_window = boardCanvas.create_window(180, 220, anchor=tkinter.NW, window=BotRight)

Score = tkinter.Button(boardCanvas, text = ("Score", Xwin, ":", Owin), command = botRightPress, state = "disabled")
Score.configure(width = 15, background = "white", activebackground = "white", relief = "flat",font=('courier',10))
Score_window = boardCanvas.create_window(50, 5, anchor=tkinter.NW, window=Score)

Reset = tkinter.Button(boardCanvas, text = "Reset", command = Reset)
Reset.configure(width = 8, background = "white", activebackground = "white", relief = "flat",font=('courier',10))
Reset_window = boardCanvas.create_window(200, 5, anchor=tkinter.NW, window=Reset)

End = tkinter.Button(boardCanvas, text = " ", command = Reset,state = "disabled")
End.configure(width = 18, height = 1, background = "#F0F0F0", activebackground = "white", relief = "flat", font=('courier',14))
End_window = boardCanvas.create_window(45, 40, anchor=tkinter.NW, window=End)

boardCanvas.pack(fill = "both", expand = 1)

game.mainloop()#start board

