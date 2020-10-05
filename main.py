import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import boxplot, xticks, ylabel, title, figure, subplot, hist, xlabel, ylim, show
import xlrd

df = pd.read_csv("heart.csv")

# extract CONTINUOUS variables
cols = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope', 'thal', 'target']

data = df[cols].to_numpy()

# X_wo includes the outliers
X_wo = data

# delete last column of X (which is the class target attribute)
X_wo = np.delete(X_wo,-1,1)

# attributeNames_wo - includes thal
attributeNames_wo = cols
attributeNames_wo = np.delete(attributeNames_wo, -1)


######### REMOVING OUTLIERS ###################################################
# We create a matrix X without the outliers.
# Outliers are removed based on the boxplot quantiles
cols2 = ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope']
for i in cols2:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1    #IQR is interquartile range. 
    filter = (df[i] >= Q1 - 1.5 * IQR) & (df[i] <= Q3 + 1.5 *IQR)
    df = df.loc[filter] 

data = df[cols].to_numpy()

X = data
# delete target column
X = np.delete(X,-1,1)

# delete thal column, due to the boxplot outlier - check data_visualization
X = np.delete(X,-1,1)

# we also update the attributeNames so it doesn't include target
attributeNames = cols
attributeNames = np.delete(attributeNames, -1)
attributeNames = np.delete(attributeNames, -1)


# define class labels based on target
# 0 - low chance of getting heart attack
# 1 - high chance of getting heart attack

classLabels_float = data[:,-1]
# replace class labels with "Less Chance for Heart Attack" = 0 and "More Chance for Heart Attack
classLabels = np.empty(classLabels_float.size, dtype=object) 
for i in range(classLabels.size):
    if classLabels_float[i] == 0:
        classLabels[i] = "Less Chance for Heart Attack"
    else:
        classLabels[i] = "More Chance for Heart Attack"

classNames = np.unique(classLabels)
classDict = dict(zip(classNames,range(len(classNames))))
y = np.array([classDict[cl] for cl in classLabels])    

# no. of objects and attributes
N, M = X.shape
# no of classes
C = len(classNames)









    