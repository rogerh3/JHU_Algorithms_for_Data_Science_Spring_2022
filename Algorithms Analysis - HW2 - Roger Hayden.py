#Roger H Hayden III
#Algorithms HW 2
#2/19/22

import pandas as pd
import numpy as np
import statistics 
from scipy import stats
import matplotlib.pylab as plt
import matplotlib.cm as cm
from scipy.fftpack import dct, idct
from sklearn.decomposition import PCA


###############################################################################
#Problem Number 1

df = pd.read_csv(r'C:\Users\roger\OneDrive\Desktop\iris.csv')
print(df)

print(df.loc[:,"sepal_length"])

#Minimum Values

print("-----------------------------------------------")

#Sepal Length
print("\n The Min Value for sepal_length is:", end = " ")
print(min(df.loc[:, "sepal_length"]))

#Sepal Width
print("\n The Min Value for sepal_width is:", end = " ")
print(min(df.loc[:, "sepal_width"]))

#Petal Length
print("\n The Min Value for petal_length is:", end = " ")
print(min(df.loc[:, "petal_length"]))

#Petal Width
print("\n The Min Value for petal_width is:", end = " ")
print(min(df.loc[:, "petal_width"]))

#Maximum Values

print("-----------------------------------------------")

#Sepal Length
print("\n The Max Value for sepal_length is:", end = " ")
print(max(df.loc[:, "sepal_length"]))

#Sepal Width
print("\n The Max Value for sepal_width is:", end = " ")
print(max(df.loc[:, "sepal_width"]))

#Petal Length
print("\n The Max Value for petal_length is:", end = " ")
print(max(df.loc[:, "petal_length"]))

#Petal Width
print("\n The Max Value for petal_width is:", end = " ")
print(max(df.loc[:, "petal_width"]))

#Mean

print("-----------------------------------------------")

#Sepal Length
print("\n The Mean for sepal_length is:", end = " ")
print(statistics.mean(df.loc[:, "sepal_length"]))

#Sepal Width
print("\n The Mean for sepal_width is:", end = " ")
print(statistics.mean(df.loc[:, "sepal_width"]))

#Petal Length
print("\n The Mean for petal_length is:", end = " ")
print(statistics.mean(df.loc[:, "petal_length"]))

#Petal Width
print("\n The Mean for petal_width is:", end = " ")
print(statistics.mean(df.loc[:, "petal_width"]))

#Trimmed Mean

print("-----------------------------------------------")

print("\nAll of the following have trimmed percent of 20")

#Sepal Length
print("\n The Trimmed Mean for sepal_length is:", end = " ")
print(stats.trim_mean(df['sepal_length'], proportiontocut=0.2))

#Sepal Width
print("\n The Trimmed Mean for sepal_width is:", end = " ")
print(stats.trim_mean(df['sepal_width'], proportiontocut=0.2))

#Petal Length
print("\n The Trimmed Mean for petal_length is:", end = " ")
print(stats.trim_mean(df['petal_length'], proportiontocut=0.2))

#Petal Width
print("\n The Trimmed Mean for petal_width is:", end = " ")
print(stats.trim_mean(df['petal_width'], proportiontocut=0.2))

#Standard Deviation

print("-----------------------------------------------")

#Sepal Length
print("\n The Standard Deviation for sepal_length is:", end = " ")
print(statistics.stdev(df.loc[:, "sepal_length"]))

#Sepal Width
print("\n The Standard Deviation for sepal_width is:", end = " ")
print(statistics.stdev(df.loc[:, "sepal_width"]))

#Petal Length
print("\n The Standard Deviation for petal_length is:", end = " ")
print(statistics.stdev(df.loc[:, "petal_length"]))

#Petal Width
print("\n The Standard Deviation for petal_width is:", end = " ")
print(statistics.stdev(df.loc[:, "petal_width"]))

#Skewness

print("-----------------------------------------------")

#Sepal Length
print("\n The Skewness for sepal_length is:", end = " ")
print(stats.skew(df.loc[:, "sepal_length"]))

#Sepal Width
print("\n The Skewness for sepal_width is:", end = " ")
print(stats.skew(df.loc[:, "sepal_width"]))

#Petal Length
print("\n The Skewness for petal_length is:", end = " ")
print(stats.skew(df.loc[:, "petal_length"]))

#Petal Width
print("\n The Skewness for petal_width is:", end = " ")
print(stats.skew(df.loc[:, "petal_width"]))

#Kurtosis

print("-----------------------------------------------")

#Sepal Length
print("\n The Kurtosis for sepal_length is:", end = " ")
print(stats.kurtosis(df.loc[:, "sepal_length"]))

#Sepal Width
print("\n The Kurtosis for sepal_width is:", end = " ")
print(stats.kurtosis(df.loc[:, "sepal_width"]))

#Petal Length
print("\n The Kurtosis for petal_length is:", end = " ")
print(stats.kurtosis(df.loc[:, "petal_length"]))

#Petal Width
print("\n The Kurtosis for petal_width is:", end = " ")
print(stats.kurtosis(df.loc[:, "petal_width"]))

###############################################################################
#Problem Number 2
df2 = pd.read_csv(r'C:\Users\roger\OneDrive\Desktop\Digit_Recognizer_train.csv')
print(df2)

df2.info()

#28 x 28 Matrix
#Used: https://richcorrado.github.io/MNIST_Digits-overview.html for assistance
def pixel_mat(row):
    # we're working with train_df so we want to drop the label column
    vec = df2.drop('label', axis=1).iloc[row].values
    # numpy provides the reshape() function to reorganize arrays into specified shapes
    pixel_mat = vec.reshape(28,28)
    return pixel_mat

x = np.random.randint(0,42000)
X = pixel_mat(x)
X

#Plot Matrix Indicies 1, 2, 4, 7, 8, 9, 11, 12, 17, and 22
#Each done individually
plt.matshow(pixel_mat(0), cmap=plt.cm.gray)
plt.title("Digit Label: %d" % df2['label'].iloc[0])
plt.show()

plt.matshow(pixel_mat(1), cmap=plt.cm.gray)
plt.title("Digit Label: %d" % df2['label'].iloc[1])
plt.show()

plt.matshow(pixel_mat(3), cmap=plt.cm.gray)
plt.title("Digit Label: %d" % df2['label'].iloc[3])
plt.show()

plt.matshow(pixel_mat(6), cmap=plt.cm.gray)
plt.title("Digit Label: %d" % df2['label'].iloc[6])
plt.show()

plt.matshow(pixel_mat(7), cmap=plt.cm.gray)
plt.title("Digit Label: %d" % df2['label'].iloc[7])
plt.show()

plt.matshow(pixel_mat(8), cmap=plt.cm.gray)
plt.title("Digit Label: %d" % df2['label'].iloc[8])
plt.show()

plt.matshow(pixel_mat(10), cmap=plt.cm.gray)
plt.title("Digit Label: %d" % df2['label'].iloc[10])
plt.show()

plt.matshow(pixel_mat(11), cmap=plt.cm.gray)
plt.title("Digit Label: %d" % df2['label'].iloc[11])
plt.show()

plt.matshow(pixel_mat(16), cmap=plt.cm.gray)
plt.title("Digit Label: %d" % df2['label'].iloc[16])
plt.show()

plt.matshow(pixel_mat(21), cmap=plt.cm.gray)
plt.title("Digit Label: %d" % df2['label'].iloc[21])
plt.show()

#My created method to do them all at once
array = [0,1,3,6,7,8,10,11,16,21]

def images(array):
    for x in array:
        plt.matshow(pixel_mat(x), cmap=plt.cm.gray)
        plt.title("Digit Label: %d" % df2['label'].iloc[x])
        plt.show()
        x+1
        
images(array)

###############################################################################
#Problem Number 3

#Part A and B
#Used: https://stackoverflow.com/questions/7110899/how-do-i-apply-a-dct-to-an-image-in-python
#for assistance
# implement 2D DCT
def dct2(a):
    return dct(dct(a, norm='ortho'), norm='ortho')

# implement 2D IDCT
def idct2(a):
    return idct(idct(a, norm='ortho'), norm='ortho')

diagMask = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0],
            [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]])

fig = plt.figure()
plt.imshow(diagMask, cmap = cm.gray)

vertMask = np.array([[0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
            [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

fig = plt.figure()
plt.imshow(vertMask, cmap = cm.gray)
       
horizMask = np.array([[0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
             [1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]])

fig = plt.figure()
plt.imshow(horizMask, cmap = cm.gray)

classlabel = df2['label']


dataD = np.array([])
dataV = np.array([])
dataH = np.array([])

df2_nolabel = df2.iloc[: , 1:]
print(df2_nolabel)

for i in range(100):
    observation = df2_nolabel.loc[i][:]
    img = observation.to_numpy().reshape(28,28)
    imgDCT = dct2(img)
    imgD = np.multiply(imgDCT,diagMask)
    imgV = np.multiply(imgDCT,vertMask)
    imgH = np.multiply(imgDCT,horizMask)
    dataD = np.concatenate((dataD, imgD[diagMask==1]), axis=0)
    dataV = np.concatenate((dataV, imgV[vertMask==1]), axis=0)
    dataH = np.concatenate((dataH, imgH[horizMask==1]), axis=0)
    
dataD = dataD.reshape((-1, imgDCT[diagMask==1].size))
dataV = dataV.reshape((-1, imgDCT[vertMask==1].size))
dataH = dataH.reshape((-1, imgDCT[horizMask==1].size))

# print(np.reshape(dataD))
# print(np.reshape(dataV))
# print(np.reshape(dataH))

#Part C - F

#For dataD
pca = PCA()
print(dataD.T.shape)
pca.fit(dataD)
c_eigVecD = pca.components_
print(c_eigVecD.T.shape)
c_eigValD = pca.explained_variance
print(c_eigValD.shape)
c_explainedD = pca.explained_variance_ratio_
print(c_explainedD)
pcaFeaturesD = np.matmul(dataD, c_eigVecD[0:15, :]).T

#For dataV
pca = PCA()
print(dataV.T.shape)
pca.fit(dataV)
c_eigVecV = pca.components_
print(c_eigVecV.T.shape)
c_eigValV = pca.explained_variance
print(c_eigValV.shape)
c_explainedV = pca.explained_variance_ratio_
print(c_explainedV)
pcaFeaturesV = np.matmul(dataV, c_eigVecV[0:10, :]).T

#For dataH
pca = PCA()
print(dataH.T.shape)
pca.fit(dataH)
c_eigVecH = pca.components_
print(c_eigVecH.T.shape)
c_eigValH = pca.explained_variance
print(c_eigValH.shape)
c_explainedH = pca.explained_variance_ratio_
print(c_explainedH)
pcaFeaturesH = np.matmul(dataH, c_eigVecH[0:10, :]).T

pcaFeatures = np.hstack(pcaFeaturesD, pcaFeaturesV)
pcaFeatures = np.hstack(pcaFeatures, pcaFeaturesH)

ft1 = 2
ft2 = 15
plt.figure()
#plt.scatter(pcaFeatures[classlabel==0, ft1], pcaFeatures[classlabel==0, ft2]), color = "red", label = 'digits0')
#plt.scatter(pcaFeatures[classlabel==1, ft1], pcaFeatures[classlabel==1, ft2]), color = "green", label = 'digits1')
#plt.scatter(pcaFeatures[classlabel==3, ft1], pcaFeatures[classlabel==3, ft2]), color = "yellow", label = 'digits3')
#plt.scatter(pcaFeatures[classlabel==4, ft1], pcaFeatures[classlabel==4, ft2]), color = "cyan", label = 'digits4')
#plt.legend(["digits0", "digits1", "digits3", "digits4"])
plt.show()

#out_to_csv(pca_output, header = False, index = False)

