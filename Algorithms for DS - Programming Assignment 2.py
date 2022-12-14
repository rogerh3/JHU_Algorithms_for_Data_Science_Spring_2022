#!/usr/bin/env python
# coding: utf-8

# In[2]:


#Roger H Hayden III
#Algorithms for DS
#Programming Assignment 2


# In[109]:


import pandas as pd
import numpy as np
import statistics 
from scipy import stats
import matplotlib.pylab as plt
import matplotlib.cm as cm
from scipy.fftpack import dct, idct

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.feature_selection import RFE
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_selection import RFECV
from sklearn.preprocessing import MinMaxScaler

from os import system
import time
from math import inf as infinity
from sklearn.decomposition import PCA

#Cross Vaildation
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score, f1_score

#Support Vector Regression
from sklearn import svm
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score


# In[2]:


df = pd.read_excel(r'C:\Users\roger\OneDrive\Desktop\trainfeatures42K.xlsx')
print(df)


# In[3]:


df


# In[4]:


#Random Sample of 5000 
df_5K = df.sample(n=5000, replace = False)
df_5K = df_5K.reset_index(drop = True)
df_5K


# In[5]:


#Smaller Sample for Running Code
df_500 = df.sample(n=500, replace = False)
df_500 = df_500.reset_index(drop = True)
df_500


# # Normalization

# In[6]:


#array = df_5K.values
#X = array[:,1:61]
#Y = array[:,0]
X = df_5K.iloc[:,1:61]
y = df_5K.iloc[:,0]

#Smaller Sample
X2 = df_500.iloc[:,1:61]
y2 = df_500.iloc[:,0]


# In[7]:


X


# In[8]:


y


# In[9]:


scaler = preprocessing.MinMaxScaler()
X_scaled = scaler.fit_transform(X)

df_norm = pd.DataFrame(X_scaled)
df_norm.insert(0, "number", y)
df_norm


# In[10]:


#Larger Sample
# X_normalized = preprocessing.normalize(X, norm = 'l2')
# X = X_normalized

# #Smaller Sample
# X2_normalized = preprocessing.normalize(X, norm = 'l2')
# X2 = X2_normalized

# X


# # Outlier Removal

# In[11]:


#find absolute value of z-score for each observation
z = np.abs(stats.zscore(df_norm))

#only keep rows in dataframe with all z-scores less than absolute value of 3 
data_clean = df_norm[(z<3).all(axis=1)]


# In[12]:


data_clean


# # Feature Ranking and Selection

# In[13]:


#Reassign X and y
X = data_clean.iloc[:,1:61]
y = data_clean.iloc[:,0]


# In[14]:


#Used this site for reference
#https://machinelearningmastery.com/feature-selection-machine-learning-python/
from sklearn.ensemble import ExtraTreesClassifier
model = ExtraTreesClassifier(n_estimators=10)
model.fit(X, y)
Features = model.feature_importances_
print(Features)


# In[15]:


#Used this site for reference
#https://www.geeksforgeeks.org/python-indices-of-numbers-greater-than-k/
res = []
for idx in range(0, len(Features)) :
    if Features[idx] > 0.01:
        res.append(idx)

print("The list of indices greater than 0.01 : " + str(res))


# In[69]:


#Selecting only the features that meet the criteria
data_clean2 = data_clean.iloc[:, res]
data_clean2.reset_index(drop = True, inplace = True)
data_clean2


# # Additional Unsuccessful Feature Testing

# In[32]:


#rfe = RFE(estimator=GradientBoostingClassifier(), n_features_to_select=60)
#rfe


# In[33]:


#model = GradientBoostingClassifier()
#model


# In[34]:


# pipe = Pipeline([("Feature Selection", rfe), ("Model", model)])
# cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=5, random_state=36851234)
# n_scores = cross_val_score(pipe, X_train, y_train, scoring="accuracy", cv=cv, n_jobs=-1)
# np.mean(n_scores)


# In[35]:


# pipe.fit(X_train, y_train)


# In[36]:


# rfe.support_


# In[37]:


# pd.DataFrame(rfe.support_,index=X.columns,columns=["Rank"])


# In[38]:


# rf_df = pd.DataFrame(rfe.ranking_,index=X.columns,columns=["Rank"]).sort_values(by="Rank",ascending=True)
# rf_df.head()


# In[39]:


# rfecv = RFECV(estimator=GradientBoostingClassifier())
# print("Optimal number of features : %d" % rfecv.n_features_)


# In[40]:


# def remove_correlated_features(X):
#     corr_threshold = 0.9
#     corr = X.corr()
#     drop_columns = np.full(corr.shape[0], False, dtype=bool)
#     for i in range(corr.shape[0]):
#         for j in range(i + 1, corr.shape[0]):
#             if corr.iloc[i, j] >= corr_threshold:
#                 drop_columns[j] = True
#     columns_dropped = X.columns[drop_columns]
#     X.drop(columns_dropped, axis=1, inplace=True)
#     return columns_dropped

# def remove_less_significant_features(X, Y):
#     sl = 0.05
#     regression_ols = None
#     columns_dropped = np.array([])
#     for itr in range(0, len(X.columns)):
#         regression_ols = sm.OLS(Y, X).fit()
#         max_col = regression_ols.pvalues.idxmax()
#         max_val = regression_ols.pvalues.max()
#         if max_val > sl:
#             X.drop(max_col, axis='columns', inplace=True)
#             columns_dropped = np.append(columns_dropped, [max_col])
#         else:
#             break
#     regression_ols.summary()
#     return columns_dropped


# In[41]:


# rfecv.support_rfecv_df = pd.DataFrame(rfecv.ranking_,index=X.columns,columns=["Rank"]).sort_values(by="Rank",ascending=True)
# rfecv_df.head()


# In[42]:


# plt.figure(figsize=(12,6))
# plt.xlabel("Number of features selected")
# plt.ylabel("Cross validation score (nb of correct classifications)")
# plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
# plt.show()


# # Dimensionality Reduction

# In[72]:


#Reassign X and y
X = data_clean2.iloc[:,1:61]
y = data_clean2.iloc[:,0]


# In[73]:


X


# In[74]:


#y.reset_index(drop = True, inplace = True)
y


# In[75]:


Best = SelectKBest(chi2, k=20).fit_transform(X, y)
Best


# In[76]:


reduction = pd.DataFrame(Best)
reduction.insert(0, "Number", y)
reduction


# # Additional Unsuccessful Dimensionality Reduction

# In[ ]:


# Failed PCA
#Used this site for reference
#https://towardsdatascience.com/pca-clearly-explained-how-when-why-to-use-it-and-feature-importance-a-guide-in-python-7c274582c37e#:~:text=PCA%20technique%20is%20particularly%20useful,for%20denoising%20and%20data%20compression.
# The PCA model
# pca = PCA(n_components=2) 
# X_new = pca.fit_transform(X) 


# In[ ]:


# fig, axes = plt.subplots(1,2)

# axes[0].scatter(X[:,0], X[:,1], c=y)
# axes[0].set_xlabel('x1')
# axes[0].set_ylabel('x2')
# axes[0].set_title('Before PCA')

# axes[1].scatter(X_new[:,0], X_new[:,1], c=y)
# axes[1].set_xlabel('PC1')
# axes[1].set_ylabel('PC2')
# axes[1].set_title('After PCA')

# plt.show()


# In[ ]:


#This does not appear to work well, however the code does run and work
#I am a little unsure as to why we aren't doing this directly on the iris
#dataset
#print(pca.explained_variance_ratio_)


# # SVM Model

# In[100]:


#Reassign X and y
X = reduction.iloc[:,1:61]
y = reduction.iloc[:, 0]
X_train, X_test, y_train, y_test = train_test_split(X, y,train_size=0.8, random_state=0)


# In[101]:


C = 1.0 #0.01
clf = svm.SVC(kernel='linear', C=C)
clf.fit(X_train, y_train)


# In[107]:


cv = cross_val_score(clf, X_train, y_train, cv=5, scoring='accuracy')
print("Average Accuracy: \t {0:.4f}".format(np.mean(cv)))
print("Accuracy SD: \t\t {0:.4f}".format(np.std(cv)))


# In[110]:


y_train_pred = cross_val_predict(clf, X_train, y_train, cv=5)
confusion_matrix(y_train, y_train_pred)


# In[111]:


print("Precision Score: \t {0:.4f}".format(precision_score(y_train, y_train_pred, average='weighted')))
print("Recall Score: \t\t {0:.4f}".format(recall_score(y_train, y_train_pred, average='weighted')))
print("F1 Score: \t\t {0:.4f}".format(f1_score(y_train, y_train_pred, average='weighted')))


# In[112]:


#Using Test Data
y_test_pred = cross_val_predict(clf, X_test, y_test, cv=5)


# In[113]:


confusion_matrix(y_test, y_test_pred)


# In[114]:


print("Precision Score: \t {0:.4f}".format(precision_score(y_test, y_test_pred, average='weighted')))
print("Recall Score: \t\t {0:.4f}".format(recall_score(y_test, y_test_pred, average='weighted')))
print("F1 Score: \t\t {0:.4f}".format(f1_score(y_test, y_test_pred, average='weighted')))


# --------------
# # Game Theory Tic Tac Toe

# In[ ]:


#I was unable to successfully figure out how to implement minimax or alpha beta algorithms myself in a few different situations
#However I was able to find a youtube video and follow along to create the following program
#https://www.youtube.com/watch?v=WA8dnbTMxyM

from math import inf
import sys, os

HUMAN = 1
COMP = -1

board = [[0, 0, 0],
         [0, 0, 0],
         [0, 0, 0]]

MSG = "Would you like to play tic tac toe? (y/n)"


def evaluate(state):
    if wins(state, COMP):
        score = -1
    elif wins(state, HUMAN):
        score = 1
    else:
        score = 0

    return score

def empty_cells(state):
    cells = [] # it contains all empty cells

    # Use enumerate for easy indexing
    for i, row in enumerate(state):
        for j, col in enumerate(row):
            if state[i][j] == 0:
                cells.append([i, j])

    return cells

def wins(state, player):
    win_state = [
        [state[0][0], state[0][1], state[0][2]],
        [state[1][0], state[1][1], state[1][2]],
        [state[2][0], state[2][1], state[2][2]],
        [state[0][0], state[1][0], state[2][0]],
        [state[0][1], state[1][1], state[2][1]],
        [state[0][2], state[1][2], state[2][2]],
        [state[0][0], state[1][1], state[2][2]],
        [state[2][0], state[1][1], state[0][2]],
    ]

    if [player, player, player] in win_state:
        return True
    else:
        return False

def game_over(state):
    return wins(state, HUMAN) or wins(state, COMP)

def clean():
    os_name = sys.platform.lower()
    os.system("cls")
    if 'win' in os_name:
        os.system('cls')
    else:
        os.system('clear')

def minimax(state, depth, player):
    if player == COMP:
        best = [-1, -1, inf] # inf/-inf are the initial score for the players
    else:
        best = [-1, -1, -inf]

    if depth == 0 or game_over(state):
        score = evaluate(state)
        return [-1, -1, score]

    for cell in empty_cells(state):
        # Fill the empty cells with the player symbols
        x, y = cell[0], cell[1]
        state[x][y] = player
        #
        score = minimax(state, depth - 1, -player)
        state[x][y] = 0
        score[0], score[1] = x, y

        if player == COMP:
            if score[2] < best[2]:
                best = score
        else:
            if score[2] > best[2]:
                best = score

    return best

def human_turn(state):
    # All possible moves
    moves = {
        1: [0, 0], 2: [0, 1], 3: [0, 2],
        4: [1, 0], 5: [1, 1], 6: [1, 2],
        7: [2, 0], 8: [2, 1], 9: [2, 2],
    }

    remain = empty_cells(state)
    isTurn = True
    print("Human Turn")
    while isTurn:
        try:
            move = int(input("Enter your move (1-9) :"))
            # When the player move is valid
            if moves.get(move) in remain:
                x, y = moves.get(move)
                state[x][y] = HUMAN
                isTurn = False

            else: # Otherwise
                print("Bad Move, try again.")

        # When the player mistype
        except ValueError:
            print("Blank spaces and string are prohibited, please enter (1-9)")

    # While-else loop, this code below will run after successful loop.
    else:
        # Clean the terminal, and show the current board
        clean()
        print(render(state))

def ai_turn(state):
    depth = len(empty_cells(state)) # The remaining of empty cells
    row, col, score = minimax(state, depth, COMP) # the optimal move for computer
    state[row][col] = COMP
    print("A.I Turn")
    print(render(state)) # Show result board

def render(state):
    legend = {0: " ", 1: "X", -1: "O"}
    state = list(map(lambda x: [legend[y] for y in x], state))
    result = "{}\n{}\n{}\n".format(*state)
    return result

def main():
    print(MSG)

    start = False
    while not start:
        confirm = input("")

        if confirm.lower() in ["y", "yes"]:
            start = True
        elif confirm.lower() in ["n", "no"]:
            sys.exit()
        else:
            print("Please enter 'y' or 'n'")

    else:
        clean()
        print("Game is settled !\n")
        print(render(board), end="\n")

    while not wins(board, COMP) and not wins(board, HUMAN):
            human_turn(board)
            if len(empty_cells(board)) == 0: break
            ai_turn(board)

    if wins(board, COMP):
        print("A.I wins!")
    elif wins(board, HUMAN):
        print("Human wins!")
    else:
        print("It's a Draw. No one wins")


if __name__ == '__main__':
    main()


# First Attempt

# In[34]:


# #Display who won and add to win counter
# def whoWin(x,End,Xwin,Owin): 
#     Xwin = 0
#     Owin = 0
#     if x == 1:
#         End.configure(text="Player 1 has won!", background = 'white')
#         Xwin = 1
#     elif x == 2:
#         End.configure(text="Player 2 has won!", background = 'white')
#         Owin = 1
#     else:
#         End.configure(text="Nobody Wins", background = 'white')
#     gameover = 1
#     L = [Xwin,Owin,gameover]
#     return L

# #Check if there is a three in a row
# #If there is a win, a display which team one and count that win
# def checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill): 
#     if place[1] == place[0] and place[0] == place[2] and place[1] != 0:
#         print ("Player",place[1]," wins")
#         return whoWin(place[1],End,Xwin,Owin)
#     if place[0] == place[3] and place[0] == place[6] and place[0] != 0:
#         print ("Player",place[0]," wins")
#         return whoWin(place[0],End,Xwin,Owin)
#     if place[0] == place[4] and place[0] == place[8] and place[0] != 0:
#         print ("Player",place[0]," wins")
#         return whoWin(place[0],End,Xwin,Owin)
#     if place[1] == place[4] and place[1] == place[7] and place[1] != 0:
#         print ("Player",place[1]," wins")
#         return whoWin(place[1],End,Xwin,Owin)
#     if place[2] == place[4] and place[2] == place[6] and place[2] != 0:
#         print ("Player",place[2]," wins")
#         return whoWin(place[2],End,Xwin,Owin)
#     if place[2] == place[5] and place[2] == place[8] and place[2] != 0:
#         print ("Player",place[2]," wins")
#         return whoWin(place[2],End,Xwin,Owin)
#     if place[3] == place[4] and place[3] == place[5] and place[3] != 0:
#         print ("Player",place[3]," wins")
#         return whoWin(place[3],End,Xwin,Owin)
#     if place[6] == place[7] and place[8] == place[6] and place[6] != 0:
#         print ("Player",place[6]," wins")
#         return whoWin(place[7],End,Xwin,Owin)
#     tie = 1
#     for i in place:
#         if i == 0:
#             tie = 0
#     if tie == 1:
#         return whoWin(3,End,Xwin,Owin)
        
#     return [0,0,0]

# #Check who won without calling whoWin
# #Necessary for MiniMax
# def checkWin2(place):
#     if place[1] == place[0] and place[0] == place[2] and place[1] != 0:
#         return place[1]
#     if place[0] == place[3] and place[0] == place[6] and place[0] != 0:
#         return place[0]
#     if place[0] == place[4] and place[0] == place[8] and place[0] != 0:
#         return place[0]
#     if place[1] == place[4] and place[1] == place[7] and place[1] != 0:
#         return place[1]
#     if place[2] == place[4] and place[2] == place[6] and place[2] != 0:
#         return place[2]
#     if place[2] == place[5] and place[2] == place[8] and place[2] != 0:
#         return place[2]
#     if place[3] == place[4] and place[3] == place[5] and place[3] != 0:
#         return place[3]
#     if place[6] == place[7] and place[8] == place[6] and place[6] != 0:
#         return place[6]
#     tie = 1
#     for i in place:
#         if i == 0:
#             tie = 0
#     if tie == 1:
#         return 0
        
#     return [0,0,0]

# #Check possibilities for wins in the next move
# def checkWinPos(place):
#     #Columns
#     if abs(place[0] + place[1] + place[2]) == 2:
#         if abs(place[0]) != 1:
#             return 0
#         elif abs(place[1]) != 1:
#             return 1
#         else:
#             return 2 
#     if abs(place[3] + place[4] + place[5]) == 2:
#         if abs(place[3]) != 1:
#             return 3
#         elif abs(place[4]) != 1:
#             return 4
#         else:
#             return 5  
#     if abs(place[6] + place[7] + place[8]) == 2:
#         if abs(place[6]) != 1:
#             return 6
#         elif abs(place[7]) != 1:
#             return 7
#         else:
#             return 8 
#     #Rows
#     if abs(place[0] + place[3] + place[6]) == 2:
#         if abs(place[0]) != 1:
#             return 0
#         elif abs(place[3]) != 1:
#             return 3
#         else:
#             return 6
#     if abs(place[1] + place[4] + place[7]) == 2:
#         if abs(place[1]) != 1:
#             return 1
#         elif abs(place[4]) != 1:
#             return 4
#         else:
#             return 7
#     if abs(place[2] + place[5] + place[8]) == 2:
#         if abs(place[2]) != 1:
#             return 2
#         elif abs(place[5]) != 1:
#             return 5
#         else:
#             return 8
#     #Diagonal
#     if abs(place[0] + place[4] + place[8]) == 2:
#         if abs(place[0]) != 1:
#             return 0
#         elif abs(place[4]) != 1:
#             return 4
#         else:
#             return 8
#     if abs(place[2] + place[4] + place[6]) == 2:
#         if abs(place[2]) != 1:
#             return 2
#         elif abs(place[4]) != 1:
#             return 4
#         else:
#             return 6


# In[35]:


# import tkinter
# import sys
# #import cv2
# #import PIL.Image, PIL.ImageTk
# import random
# sys.setrecursionlimit(2000)#Add limit for recursion
# from checkWin_Incomplete import checkWin
# #from checkWin_Incomplete import checkWin2
# from checkWin_Incomplete import checkWinPos
# game = tkinter.Toplevel()#init board
# game.geometry("350x400+300+300") #set base dimensions


# boardCanvas = tkinter.Canvas(game, width = 640, height = 640)#Initialize TKinter canvas

# AIturn = 0 #AI goes second
# #boardImage = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(board))  #Load Board
# boardCanvas.create_image(50, 89,anchor= tkinter.NW)
# boardCanvas.create_line(114, 90, 114, 280)
# boardCanvas.create_line(173, 90, 173, 280)
# boardCanvas.create_line(50, 153, 240, 153)
# boardCanvas.create_line(50, 215, 240, 215)
# aiSkill = 1 #Change to 0 for no AI, Change to 1 for Easy, Change to 2 for Medium, 3 for Hard
# # Easy follows optimal move pattern if no places are marked
# # Medium is easy with additional conditional statements
# # Hard uses the MiniMax algorithm
# aiX = 0
# def decisionMaker(boardState,minimax,depth):
#     while True: #Delete this loop if attempting MiniMax
#         c = random.randint(0,8)
#         if boardState[c] == 0:
#             return c
#     #Do MiniMax Here
    
# def getPlace():
#     global place
#     return place


# def AI(aiSkill,place,turn,AIturn):
#     AIturn = AIturn+1 #Alternate player and AI
#     if aiSkill == 1: #Easy
#         if place[4] == 0:
#             midMidPress()
#         elif place[0] == 0:
#             topLeftPress()
#         elif place[2] == 0:
#             botLeftPress()
#         elif place[6] == 0:
#             topRightPress()
#         elif place[8] == 0:
#             botRightPress()
#         elif place[1] == 0:
#             midLeftPress()
#         elif place[3] == 0:
#             topMidPress()
#         elif place[5] == 0:
#             botMidPress()
#         elif place[7] == 0:
#             midRightPress()
#     if aiSkill == 2: #Medium
#         F = checkWinPos(place)
#         if F != None:
#             if F == 0:
#                 topLeftPress()
#             if F == 1:
#                 midLeftPress()
#             if F == 2:
#                 botLeftPress()
#             if F == 3:
#                 topMidPress()
#             if F == 4:
#                 midMidPress()
#             if F == 5:
#                 botMidPress()
#             if F == 6:
#                 topRightPress()
#             if F == 7:
#                 midRightPress()
#             if F == 8:
#                 botRightPress()
#         elif place[4] == 0:
#             midMidPress()
#         elif place[0] == 0:
#             topLeftPress()
#         elif place[2] == 0:
#             botLeftPress()
#         elif place[6] == 0:
#             topRightPress()
#         elif place[8] == 0:
#             botRightPress()
#         elif place[1] == 0:
#             midLeftPress()
#         elif place[3] == 0:
#             topMidPress()
#         elif place[5] == 0:
#             botMidPress()
#         elif place[7] == 0:
#             midRightPress()
#     if aiSkill == 3: #Hard
#         G = [] #Create new list G. If G = place, python thinks it is actually place under a different name.
#         for i in place:
#             G.append(i)

#         F = decisionMaker(G,0,0)
#         if F == 0:
#             topLeftPress()
#         if F == 1:
#             midLeftPress()
#         if F == 2:
#             botLeftPress()
#         if F == 3:
#             topMidPress()
#         if F == 4:
#             midMidPress()
#         if F == 5:
#             botMidPress()
#         if F == 6:
#             topRightPress()
#         if F == 7:
#             midRightPress()
#         if F == 8:
#             botRightPress()
            
# #initialize filled places and whose turn it is
# place = [0,0,0,0,0,0,0,0,0]
# turn = 0

# #Initialize variables to describe whether boxes are marked or not
# topLeftComp = 0
# midLeftComp = 0
# botLeftComp = 0
# topMidComp = 0
# midMidComp = 0
# botMidComp = 0
# topRightComp = 0
# midRightComp = 0
# botRightComp = 0

# #Initialize variables for after the game is completed. 
# #Counts wins by each player and stops moves when game is over.
# gameOver=0
# Xwin = 0
# Owin = 0

# L = []
# #button press functions
# def topLeftPress():
#     global gameOver #No moves can be made if game is over.
#     if gameOver == 0:
#         global turn #These are required to call the function in the checkWin.py
#         global topLeftComp
#         global AIturn
#         global Xwin
#         global Owin
#         global aiSkill
#         if topLeftComp == 0: #check if space is filled   
#             if turn == 0:
#                 place[0] = 1
#                 turn = 1
#                 TopLeft.configure(text=("X"))
#             else:
#                 turn = 0
#                 place[0] = -1
#                 TopLeft.configure(text="O")
#             topLeftComp = 1
#             L = checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill)
#             Xwin = Xwin+L[0]
#             Owin = Owin+L[1]
#             gameOver = L[2]
#             if AIturn%2 == 0: #Call AI turn every two turns
#                 AIturn = AIturn+1
#                 AI(aiSkill,place,turn,AIturn)
                
#             else:
#                 AIturn = AIturn+1
#         else:
#             print("Already Set Box")
#         return place
# def midLeftPress():
#     global gameOver
#     if gameOver == 0:
#         global turn
#         global midLeftComp
#         global AIturn
#         global Xwin
#         global Owin
#         global aiSkill
#         if midLeftComp == 0:    
#             if turn == 0:
#                 place[1] = 1
#                 turn = 1
#                 MidLeft.configure(text="X")
#             else:
#                 turn = 0
#                 place[1] = -1
#                 MidLeft.configure(text="O")
#             midLeftComp = 1
#             L = checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill)
            
#             Xwin = Xwin+L[0]
#             Owin = Owin+L[1]
#             gameOver = L[2]
#             if AIturn%2 == 0:
#                 AIturn = AIturn+1
#                 AI(aiSkill,place,turn,AIturn)
                
#             else:
#                 AIturn = AIturn+1
#         else:
#             print("Already Set Box")
#         return place
# def botLeftPress():
#     global gameOver
#     if gameOver == 0:
#         global turn
#         global botLeftComp
#         global AIturn
#         global Xwin
#         global Owin
#         global aiSkill
#         if botLeftComp == 0:    
#             if turn == 0:
#                 place[2] = 1
#                 turn = 1
#                 BotLeft.configure(text="X")
#             else:
#                 turn = 0
#                 place[2] = -1
#                 BotLeft.configure(text="O")
#             botLeftComp = 1
#             L = checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill)
#             Xwin = Xwin+L[0]
#             Owin = Owin+L[1]
#             gameOver = L[2]
#             if AIturn%2 == 0:
#                 AIturn = AIturn+1
#                 AI(aiSkill,place,turn,AIturn)
                
#             else:
#                 AIturn = AIturn+1
#         else:
#             print("Already Set Box")
#         return place
# def topMidPress():
#     global gameOver
#     if gameOver == 0:
#         global turn
#         global topMidComp
#         global AIturn
#         global Xwin
#         global Owin
#         global aiSkill
#         if topMidComp == 0:    
#             if turn == 0:
#                 place[3] = 1
#                 turn = 1
#                 TopMid.configure(text="X")
#             else:
#                 turn = 0
#                 place[3] = -1
#                 TopMid.configure(text="O")
#             topMidComp = 1
#             L = checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill)
#             Xwin = Xwin+L[0]
#             Owin = Owin+L[1]
#             gameOver = L[2]
#             if AIturn%2 == 0:
#                 AIturn = AIturn+1
#                 AI(aiSkill,place,turn,AIturn)
                
#             else:
#                 AIturn = AIturn+1
#         else:
#             print("Already Set Box")
#         return place
# def midMidPress():
#     global gameOver
#     if gameOver == 0:
#         global turn
#         global midMidComp
#         global AIturn
#         global Xwin
#         global Owin
#         global aiSkill
#         if midMidComp == 0:    
#             if turn == 0:
#                 place[4] = 1
#                 turn = 1
#                 MidMid.configure(text="X")
#             else:
#                 turn = 0
#                 place[4] = -1
#                 MidMid.configure(text="O")
#             midMidComp = 1
#             L = checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill)
#             Xwin = Xwin+L[0]
#             Owin = Owin+L[1]
#             gameOver = L[2]
#             if AIturn%2 == 0:
#                 AIturn = AIturn+1
#                 AI(aiSkill,place,turn,AIturn)
                
#             else:
#                 AIturn = AIturn+1
#         else:
#             print("Already Set Box")
#         return place
# def botMidPress():
#     global gameOver
#     if gameOver == 0:
#         global turn
#         global botMidComp
#         global AIturn
#         global Xwin
#         global Owin
#         global aiSkill
#         if botMidComp == 0:    
#             if turn == 0:
#                 place[5] = 1
#                 turn = 1
#                 BotMid.configure(text="X")
#             else:
#                 turn = 0
#                 place[5] = -1
#                 BotMid.configure(text="O")
#             botMidComp = 1
#             L = checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill)
#             Xwin = Xwin+L[0]
#             Owin = Owin+L[1]
#             gameOver = L[2]
#             if AIturn%2 == 0:
#                 AIturn = AIturn+1
#                 AI(aiSkill,place,turn,AIturn)
                
#             else:
#                 AIturn = AIturn+1
#         else:
#             print("Already Set Box")
#         return place
# def topRightPress():
#     global gameOver
#     if gameOver == 0:
#         global turn
#         global topRightComp
#         global AIturn
#         global Xwin
#         global Owin
#         global aiSkill
#         if topRightComp == 0:    
#             if turn == 0:
#                 place[6] = 1
#                 turn = 1
#                 TopRight.configure(text="X")
#             else:
#                 turn = 0
#                 place[6] = -1
#                 TopRight.configure(text="O")
#             topRightComp = 1
#             L = checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill)
#             Xwin = Xwin+L[0]
#             Owin = Owin+L[1]
#             gameOver = L[2]
#             if AIturn%2 == 0:
#                 AIturn = AIturn+1
#                 AI(aiSkill,place,turn,AIturn)
                
#             else:
#                 AIturn = AIturn+1
#         else:
#             print("Already Set Box")
#         return place
# def midRightPress():
#     global gameOver
#     if gameOver == 0:
#         global turn
#         global midRightComp
#         global AIturn
#         global Xwin
#         global Owin
#         global aiSkill
#         if midRightComp == 0:    
#             if turn == 0:
#                 place[7] = 1
#                 turn = 1
#                 MidRight.configure(text="X")
#             else:
#                 turn = 0
#                 place[7] = -1
#                 MidRight.configure(text="O")
#             midRightComp = 1
#             L = checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill)
#             Xwin = Xwin+L[0]
#             Owin = Owin+L[1]
#             gameOver = L[2]
#             if AIturn%2 == 0:
#                 AIturn = AIturn+1
#                 AI(aiSkill,place,turn,AIturn)
                
#             else:
#                 AIturn = AIturn+1
#         else:
#             print("Already Set Box")
#         return place
# def botRightPress():
#     global gameOver
#     if gameOver == 0:
#         global turn
#         global botRightComp
#         global AIturn
#         global Xwin
#         global Owin
#         global aiSkill
#         if botRightComp == 0:    
#             if turn == 0:
#                 place[8] = 1
#                 turn = 1
#                 BotRight.configure(text="X")
#             else:
#                 turn = 0
#                 place[8] = -1
#                 BotRight.configure(text="O")
#             botRightComp = 1
#             L = checkWin(place,AIturn,End,Xwin,Owin,turn, aiSkill)
#             Xwin = Xwin+L[0]
#             Owin = Owin+L[1]
#             gameOver = L[2]
#             if AIturn%2 == 0:
#                 AIturn = AIturn+1
#                 AI(aiSkill,place,turn,AIturn)
                
#             else:
#                 AIturn = AIturn+1
#         else:
#             print("Already Set Box")
#         return place

# #reset all board variables
# def Reset():
#     global place
#     global turn
#     global topLeftComp
#     global midLeftComp
#     global botLeftComp
#     global topMidComp
#     global midMidComp
#     global botMidComp
#     global topRightComp
#     global midRightComp
#     global botRightComp
#     global gameOver
#     global AIturn
#     AIturn = 0
#     place = [0,0,0,0,0,0,0,0,0]
#     turn = 0
#     topLeftComp = 0
#     midLeftComp = 0
#     botLeftComp = 0
#     topMidComp = 0
#     midMidComp = 0
#     botMidComp = 0
#     topRightComp = 0
#     midRightComp = 0
#     botRightComp = 0
#     TopLeft.configure(text=" ")
#     MidLeft.configure(text=" ")
#     BotLeft.configure(text=" ")
#     TopMid.configure(text=" ")
#     MidMid.configure(text=" ")
#     BotMid.configure(text=" ")
#     TopRight.configure(text=" ")
#     MidRight.configure(text=" ")
#     BotRight.configure(text=" ")
#     End.configure(width = 18, height = 1, background = "#F0F0F0", activebackground = "white", relief = "flat", font=('courier',14),text = " ")
#     gameOver = 0
#     Score.configure(text = ("Score", Xwin, ":", Owin))
    

# #Button Setup
# TopLeft = tkinter.Button(boardCanvas, text = " ", command = topLeftPress)
# TopLeft.configure(width = 7, height = 3, background = "#F0F0F0", activebackground = "#F0F0F0", relief = "flat")
# TopLeft_window = boardCanvas.create_window(50, 90, anchor=tkinter.NW, window=TopLeft)

# MidLeft = tkinter.Button(boardCanvas, text = " ", command = midLeftPress)
# MidLeft.configure(width = 7, height = 3, background = "#F0F0F0", activebackground = "#F0F0F0", relief = "flat")
# MidLeft_window = boardCanvas.create_window(50, 158, anchor=tkinter.NW, window=MidLeft)

# BotLeft = tkinter.Button(boardCanvas, text = " ", command = botLeftPress)
# BotLeft.configure(width = 7, height = 3, background = "#F0F0F0", activebackground = "#F0F0F0", relief = "flat")
# BotLeft_window = boardCanvas.create_window(50, 220, anchor=tkinter.NW, window=BotLeft)

# TopMid = tkinter.Button(boardCanvas, text = " ", command = topMidPress)
# TopMid.configure(width = 6, height = 3, background = "#F0F0F0", activebackground = "#F0F0F0", relief = "flat")
# TopMid_window = boardCanvas.create_window(118, 90, anchor=tkinter.NW, window=TopMid)

# MidMid = tkinter.Button(boardCanvas, text = " ", command = midMidPress)
# MidMid.configure(width = 6, height = 3, background = "#F0F0F0", activebackground = "#F0F0F0", relief = "flat")
# MidMid_window = boardCanvas.create_window(118, 158, anchor=tkinter.NW, window=MidMid)

# BotMid = tkinter.Button(boardCanvas, text = " ", command = botMidPress)
# BotMid.configure(width = 6, height = 3, background = "#F0F0F0", activebackground = "#F0F0F0", relief = "flat")
# BotMid_window = boardCanvas.create_window(118, 224, anchor=tkinter.NW, window=BotMid)

# TopRight = tkinter.Button(boardCanvas, text = " ", command = topRightPress)
# TopRight.configure(width = 8, height = 3, background = "#F0F0F0", activebackground = "#F0F0F0", relief = "flat")
# TopRight_window = boardCanvas.create_window(180, 90, anchor=tkinter.NW, window=TopRight)

# MidRight = tkinter.Button(boardCanvas, text = " ", command = midRightPress)
# MidRight.configure(width = 8, height = 3, background = "#F0F0F0", activebackground = "#F0F0F0", relief = "flat")
# MidRight_window = boardCanvas.create_window(180, 158, anchor=tkinter.NW, window=MidRight)

# BotRight = tkinter.Button(boardCanvas, text = " ", command = botRightPress)
# BotRight.configure(width = 8, height = 3, background = "#F0F0F0", activebackground = "#F0F0F0", relief = "flat")
# BotRight_window = boardCanvas.create_window(180, 220, anchor=tkinter.NW, window=BotRight)

# Score = tkinter.Button(boardCanvas, text = ("Score", Xwin, ":", Owin), command = botRightPress, state = "disabled")
# Score.configure(width = 15, background = "white", activebackground = "white", relief = "flat",font=('courier',10))
# Score_window = boardCanvas.create_window(50, 5, anchor=tkinter.NW, window=Score)

# Reset = tkinter.Button(boardCanvas, text = "Reset", command = Reset)
# Reset.configure(width = 8, background = "white", activebackground = "white", relief = "flat",font=('courier',10))
# Reset_window = boardCanvas.create_window(200, 5, anchor=tkinter.NW, window=Reset)

# End = tkinter.Button(boardCanvas, text = " ", command = Reset,state = "disabled")
# End.configure(width = 18, height = 1, background = "#F0F0F0", activebackground = "white", relief = "flat", font=('courier',14))
# End_window = boardCanvas.create_window(45, 40, anchor=tkinter.NW, window=End)

# boardCanvas.pack(fill = "both", expand = 1)

# game.mainloop()#start board


# Second Attempt

# In[ ]:


# board = ["_","_","_",
#          "_","_","_",
#          "_","_","_"]

# def printBoard(board):
#     print(" ", board[0], "|", board[1], "|", board[2], " ")
#     print("---------------")
#     print(" ", board[3], "|", board[4], "|", board[5], " ")
#     print("---------------")
#     print(" ", board[6], "|", board[7], "|", board[8], " ")
    
# def evaluate(state):
#     if winner(state, "O"):
#         score = +1
#     elif winner(state, "X"):
#         score = -1
#     else:
#         score = 0
#     return score

# def availableSpots(board):
#     result = []
#     for i, j in enumerate(board):
#         if j == "_":
#             result.append(i)
            
# def winner(state, player):
#     win_state[
#         [state[0], state[1], state[2]],
#         [state[3], state[4], state[5]],
#         [state[6], state[7], state[8]],
#         [state[0], state[3], state[6]],
#         [state[1], state[4], state[7]],
#         [state[2], state[5], state[8]],
#         [state[0], state[4], state[8]],
#         [state[2], state[4], state[6]],
#             ]
#     if [player, player, player] in win_state:
#         return True
#     else:
#         return False
    
# def game_over(state):
#     return winner(state, "X") or winner(state, "O")

# def minimax(board, depth, player):
#     if player == "O":
#         best = [-1, -infinity]
#     else:
#         best = [-1, infinity]
        
#     if depth == 0 or game_over(board):
#         score = evaluate(board)
#         return[-1, score]
    
#     for cell in availableSpots(board):
#         board[cell] = player
        
#         if player == "O":
#             score = minimax(board, depth - 1, "X")
#         else:
#             score = minimax(board, depth - 1, "O")
        
#         board[cell] = "_"
#         score[0] = cell
        
#         if player == "O":
#             if best[1] < score[1]:
#                 best = score
#         else:
#             if best [1] > score[1]:
#                 best = score
                
#     return best

# def human_turn(board):
#     depth = len(availableSpots(board))
#     if depth == 0 or game_over(board):
#         return
    
#     move = -1
    
#     while move < 1 or move > 9:
#         clean()
#         print("Human Turn \n")
#         printBoard(board)
#         move = int(input("Enter position (1...9: "))
        
#         if move <= 9 or move >= 1:
#             if board[move - 1] == "_":
#                 move -= 1
#                 board[move] = "X"
#                 printBoard(board)
#                 return
#             else:
#                 print("Bad Move")
#                 move = -1
                
# def clean():
#     system('cls')
    
# def AI_Move():
#     depth = len(availableSpots(board))
#     if depth == 0 or game_over(board):
#         return
    
#     clean()
    
#     print("AI Turn \n")
#     move = minimax(board, depth, "O")
#     board[move[0]] = "O"
#     printBoard(board)
#     time.sleep(1)
    
# def main(board):
#     while len(availableSpots(board)) > 0 and not game_over(board):
#         human_turn(board)
#         AI_Move()
        
#     if winner(board, "X"):
#         print("Human Won!")
#         return 0
    
#     elif winner(board, "O"):
#         print("AI Won!")
#         return 0
#     else:
#         print("Draw")
#         return 0
    
# if __name__ == "__main__":
#     while True:
#         main(board)
#         board = ["_","_","_","_","_","_","_","_","_"]
#         again = input("Wanna play again? [y/n]: ")
#         if again == "n":
#             break


# In[2]:


# import math
# import random


# class Player():
#     def __init__(self, letter):
#         self.letter = letter

#     def get_move(self, game):
#         pass


# class HumanPlayer(Player):
#     def __init__(self, letter):
#         super().__init__(letter)

#     def get_move(self, game):
#         valid_square = False
#         val = None
#         while not valid_square:
#             square = input(self.letter + '\'s turn. Input move (0-9): ')
#             try:
#                 val = int(square)
#                 if val not in game.available_moves():
#                     raise ValueError
#                 valid_square = True
#             except ValueError:
#                 print('Invalid square. Try again.')
#         return val


# class RandomComputerPlayer(Player):
#     def __init__(self, letter):
#         super().__init__(letter)

#     def get_move(self, game):
#         square = random.choice(game.available_moves())
#         return square


# class SmartComputerPlayer(Player):
#     def __init__(self, letter):
#         super().__init__(letter)

#     def get_move(self, game):
#         if len(game.available_moves()) == 9:
#             square = random.choice(game.available_moves())
#         else:
#             square = self.minimax(game, self.letter)['position']
#         return square

#     def minimax(self, state, player):
#         max_player = self.letter  # yourself
#         other_player = 'O' if player == 'X' else 'X'

#         # first we want to check if the previous move is a winner
#         if state.current_winner == other_player:
#             return {'position': None, 'score': 1 * (state.num_empty_squares() + 1) if other_player == max_player else -1 * (
#                         state.num_empty_squares() + 1)}
#         elif not state.empty_squares():
#             return {'position': None, 'score': 0}

#         if player == max_player:
#             best = {'position': None, 'score': -math.inf}  # each score should maximize
#         else:
#             best = {'position': None, 'score': math.inf}  # each score should minimize
#         for possible_move in state.available_moves():
#             state.make_move(possible_move, player)
#             sim_score = self.minimax(state, other_player)  # simulate a game after making that move

#             # undo move
#             state.board[possible_move] = ' '
#             state.current_winner = None
#             sim_score['position'] = possible_move  # this represents the move optimal next move

#             if player == max_player:  # X is max player
#                 if sim_score['score'] > best['score']:
#                     best = sim_score
#             else:
#                 if sim_score['score'] < best['score']:
#                     best = sim_score
#         return best


# In[ ]:


# from player import HumanPlayer, RandomComputerPlayer, SmartComputerPlayer


# class TicTacToe():
#     def __init__(self):
#         self.board = self.make_board()
#         self.current_winner = None

#     @staticmethod
#     def make_board():
#         return [' ' for _ in range(9)]

#     def print_board(self):
#         for row in [self.board[i*3:(i+1) * 3] for i in range(3)]:
#             print('| ' + ' | '.join(row) + ' |')

#     @staticmethod
#     def print_board_nums():
#         # 0 | 1 | 2
#         number_board = [[str(i) for i in range(j*3, (j+1)*3)] for j in range(3)]
#         for row in number_board:
#             print('| ' + ' | '.join(row) + ' |')

#     def make_move(self, square, letter):
#         if self.board[square] == ' ':
#             self.board[square] = letter
#             if self.winner(square, letter):
#                 self.current_winner = letter
#             return True
#         return False

#     def winner(self, square, letter):
#         # check the row
#         row_ind = math.floor(square / 3)
#         row = self.board[row_ind*3:(row_ind+1)*3]
#         # print('row', row)
#         if all([s == letter for s in row]):
#             return True
#         col_ind = square % 3
#         column = [self.board[col_ind+i*3] for i in range(3)]
#         # print('col', column)
#         if all([s == letter for s in column]):
#             return True
#         if square % 2 == 0:
#             diagonal1 = [self.board[i] for i in [0, 4, 8]]
#             # print('diag1', diagonal1)
#             if all([s == letter for s in diagonal1]):
#                 return True
#             diagonal2 = [self.board[i] for i in [2, 4, 6]]
#             # print('diag2', diagonal2)
#             if all([s == letter for s in diagonal2]):
#                 return True
#         return False

#     def empty_squares(self):
#         return ' ' in self.board

#     def num_empty_squares(self):
#         return self.board.count(' ')

#     def available_moves(self):
#         return [i for i, x in enumerate(self.board) if x == " "]


# def play(game, x_player, o_player, print_game=True):

#     if print_game:
#         game.print_board_nums()

#     letter = 'X'
#     while game.empty_squares():
#         if letter == 'O':
#             square = o_player.get_move(game)
#         else:
#             square = x_player.get_move(game)
#         if game.make_move(square, letter):

#             if print_game:
#                 print(letter + ' makes a move to square {}'.format(square))
#                 game.print_board()
#                 print('')

#             if game.current_winner:
#                 if print_game:
#                     print(letter + ' wins!')
#                 return letter  # ends the loop and exits the game
#             letter = 'O' if letter == 'X' else 'X'  # switches player

#         time.sleep(.8)

#     if print_game:
#         print('It\'s a tie!')



# if __name__ == '__main__':
#     x_player = SmartComputerPlayer('X')
#     o_player = HumanPlayer('O')
#     t = TicTacToe()
#     play(t, x_player, o_player, print_game=True)

