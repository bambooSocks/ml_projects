import pandas as pd
import numpy as np

df = pd.read_csv("heart.csv")

# extract CONTINUOUS variables
cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak', 'target']

data = df[cols].to_numpy()

X = data
attributeNames = cols


# delete last column of X (which is the class attribute)
X = np.delete(X, -1, 1)

# we also update the attributeNames so it doesn't include target
attributeNames = np.delete(attributeNames, -1)

# define class labels based on target
# 0 - low chance of getting heart attack
# 1 - high chance of getting heart attack

classLabels_float = data[:, -1]
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

