import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv("heart.csv")

# extract CONTINUOUS variables
cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']



#### Removing outliers (based on boxplots from data_visualization)
for i in cols:
    Q1 = df[i].quantile(0.25)
    Q3 = df[i].quantile(0.75)
    IQR = Q3 - Q1    #IQR is interquartile range. 
    filter = (df[i] >= Q1 - 1.5 * IQR) & (df[i] <= Q3 + 1.5 *IQR)
    df = df.loc[filter] 

###

data = df[cols].to_numpy()

X_cont = data
attributeNames_cont = cols


# delete last column of X (which is the class attribute)
X_cont = np.delete(X_cont, -1, 1)

# we also update the attributeNames so it doesn't include target
attributeNames_cont = np.delete(attributeNames_cont, -1)

# define class labels based on target
# 0 - low chance of getting heart attack
# 1 - high chance of getting heart attack

classLabels_float = data[:, -1]
# replace class labels with "Less Chance for Heart Attack" = 0 and "More Chance for Heart Attack
classLabels_cont = np.empty(classLabels_float.size, dtype=object)
for i in range(classLabels_cont.size):
    if classLabels_float[i] == 0:
        classLabels_cont[i] = "Less Chance for Heart Attack"
    else:
        classLabels_cont[i] = "More Chance for Heart Attack"

classNames = np.unique(classLabels_cont)
classDict = dict(zip(classNames,range(len(classNames))))
y = np.array([classDict[cl] for cl in classLabels_cont])

# no. of objects and attributes
N_cont, M_cont = X_cont.shape
# no of classes
C = len(classNames)


# plt.boxplot(X_cont)
# plt.xticks(range(1, 5), attributeNames_cont, fontsize=8)
# plt.title('Boxplot for all attributes')
# plt.show()

# Standardized data:


