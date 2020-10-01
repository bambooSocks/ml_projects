import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import boxplot, xticks, ylabel, title, figure, subplot, hist, xlabel, ylim, show
import xlrd

df = pd.read_csv("heart.csv")
cols = np.asarray(df.columns)

data = df.to_numpy()

print(cols)
print(data)

thal = np.transpose(data)[12]

#plt.hist(thal, bins=4, rwidth=0.75, color=['red', 'blue', 'green', 'yellow'])
#plt.show()

# Data selecting subset
#re-arrange columns (so the ones we don't need are last)
X = data
attributeNames = cols
rem_col = np.array([3,4,4,5,7])
for i in range(5):
    p=rem_col[i]
    X = np.delete(X,p,1)
    attributeNames = np.delete(attributeNames, p)

# delete last column of X (which is the class attribute)
X = np.delete(X,-1,1)

#we also update the attributeNames so it doesn't include target
attributeNames = np.delete(attributeNames, -1)

# define class labels based on target
# 0 - low chance of getting heart attack
# 1 - high chance of getting heart attack

classLabels_float = data[:,-1]
#replace class labels with "Less Chance for Heart Attack" = 0 and "More Chance for Heart Attack 
classLabels = np.empty(classLabels_float.size, dtype=object) 
for i in range(classLabels.size):
    if (classLabels_float[i] == 0):
        classLabels[i] = "Less Chance for Heart Attack"
    else:
        classLabels[i] = "More Chance for Heart Attack"

classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
y = np.array([classDict[cl] for cl in classLabels])    

#no. of objects and attributes
N, M = X.shape
#no of classes
C = len(classNames)

# DATA VIZUALIZATION ###############################################

#Histograms  
figure(figsize=(8,7))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    subplot(u,v,i+1)
    hist(X[:,i], color=(0.1, 0.8-i*0.1, 0.4))
    xlabel(attributeNames[i])
    ylim(0,N/2)
    
show()

# Seems like age, chol and thalach are normally distributed

# 

# Boxplots: only for continuous data, excluding age

X_cont = np.column_stack((X[:,3],X[:,4],X[:,5]))

attributeNames_cont = np.array([attributeNames[3],attributeNames[4],attributeNames[5]])

boxplot(X_cont)
xticks(range(1,3),attributeNames_cont)
title('Boxplot for continuous values')
show()

'''
It can clearly be seen that tje continuous variables are at different scales
(e.g.: cholesterol is between 100-400 whereas olpeak is from 0 to 6)
This will be a problem for applying ML algorithms, 
especially for Classification problems where Euclidian distance is used


We will substract the mean and divide by st. dev for the continous variables:
chol, thalach, oldpeak

Note: Age can be left out of this since we might transform it into a discrete variable
later - for making a better model - for example we could use bining but this is beyond the scope
of this project.

'''


Y2 = X_cont - np.ones((N, 1))*X_cont.mean(0)
Y2 = Y2*(1/np.std(Y2,0))

# We do the boxplot again: 
boxplot(Y2)
xticks(range(1,3),attributeNames_cont)
title('Boxplot for Standardized Continuous Values')
show()

# Looks better now. We should discuss which outliers to remove

# Boxplot based on each class: more chance/ less chance of heart attack
# Also align them on the same axis for better comparison

figure(figsize=(14,7))
for c in range(C):
    subplot(1,C,c+1)
    class_mask = (y==c)     
    boxplot(Y2[class_mask,:])
    #title('Class: {0}'.format(classNames[c]))
    title('Boxplot for Class: '+classNames[c])
    xticks(range(1,len(attributeNames_cont)+1), [a[:7] for a in attributeNames_cont], rotation=45)
    y_up = Y2.max()+(Y2.max()-Y2.min())*0.1; y_down = Y2.min()-(Y2.max()-Y2.min())*0.1
    ylim(y_down, y_up)

show()

# Replace standardized columns into X
X_stand = X
X_stand[:,3] = Y2[:,0]
X_stand[:,3] = Y2[:,1]
X_stand[:,3] = Y2[:,2]



    