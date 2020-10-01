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





    