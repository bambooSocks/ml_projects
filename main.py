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

# Boxplots: 

boxplot(X)
xticks(range(1,9),attributeNames)
title('Data Set on Heart Attack Possibility - boxplot')
show()

'''
Boxplots are inequal: X should go through feature transformation.
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
X_col_tonorm = np.column_stack((X[:,3],X[:,4],X[:,5]))

Y2_col = X_col_tonorm - np.ones((N, 1))*X_col_tonorm.mean(0)
Y2_col = Y2_col*(1/np.std(Y2_col,0))

# Add columns in the matrix:
Y2 = X
Y2[:,3] = Y2_col[:,0]
Y2[:,4] = Y2_col[:,1]
Y2[:,5] = Y2_col[:,2]

# and we remove the age column:
Y2 = np.delete(Y2,0,1)    

# We do the boxplot again: 
boxplot(Y2)
attributeNames_boxplot = np.delete(attributeNames,0)
xticks(range(1,8),attributeNames_boxplot)
title('H.A.P. - Boxplot for Standardized Values')
show()

# Looks better now. We should discuss which outliers to remove

# Make a boxplot based on each class: high risk/ low risk of heart attack
# Also align them on the same axis for better comparison



figure(figsize=(14,7))
for c in range(C):
    subplot(1,C,c+1)
    class_mask = (y==c)     
    boxplot(Y2[class_mask,:])
    #title('Class: {0}'.format(classNames[c]))
    title('Class: '+classNames[c])
    xticks(range(1,len(attributeNames)+1), [a[:7] for a in attributeNames], rotation=45)
    y_up = Y2.max()+(Y2.max()-Y2.min())*0.1; y_down = Y2.min()-(Y2.max()-Y2.min())*0.1
    ylim(y_down, y_up)

show()



    