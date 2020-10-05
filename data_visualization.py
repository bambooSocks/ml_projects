# -*- coding: utf-8 -*-

from main import *
from continuous_data import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import zscore

# DATA VIZUALIZATION ###############################################



# Try histogram for all variables
figure(figsize=(8,7))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    subplot(u,v,i+1)
    hist(X[:,i], color=(0.2, 0.9-i*0.1, 0.4))
    xlabel(attributeNames[i])
    ylim(0,N/2)
    
show()


'''
Seems like age, trestrbps, chol and thalach are normally distributed
'''

# BOXPLOT: for all attributes ################################################

plt.boxplot(X_wo)
plt.xticks(range(1, 10), attributeNames_wo, fontsize=8)
plt.title('Boxplot with outliers')
plt.show()

plt.boxplot(X)
plt.xticks(range(1, 10), attributeNames, fontsize=8)
plt.title('Boxplot without outliers')
plt.show()

'''
It can clearly be seen that tje continuous variables are at different scales
(e.g.: cholesterol is between 100-400 whereas olpeak is from 0 to 6)
This will be a problem for applying ML algorithms, 
especially for Classification problems where Euclidian distance is used

In order to see the outliers more clearly we have extracted the continuos
attributes including outliers in X_cont_wo
Those attributes are: age, trestbps, chol, thalach, oldpeak
'''
# create matrix of continuous variables: 
X_cont_wo = np.column_stack((X_wo[:, 0], X_wo[:, 3], X_wo[:, 4], X_wo[:, 5], X_wo[:, 6]))

# standardize X_cont (relative to mean and standard deviation)
Y2_wo = zscore(X_cont_wo, ddof=1)

N2, M2 = X_cont_wo.shape

# Replace in X_wo the columns with standardized attrbiutes
X_stand_wo = X_wo

# Replace standardized columns into X
X_stand_wo[:, 0] = Y2_wo[:, 0]
X_stand_wo[:, 3] = Y2_wo[:, 1]
X_stand_wo[:, 4] = Y2_wo[:, 2]
X_stand_wo[:, 5] = Y2_wo[:, 3]
X_stand_wo[:, 6] = Y2_wo[:, 4]

'''
Now we make a couple of boxplots for better understanding of our data
'''

# BOXPLOT: all X dataset, with standardized continuous attributes:
plt.boxplot(X_stand_wo)
plt.xticks(range(1, 10), attributeNames_wo, fontsize=8)
plt.title('Boxplot to check for outliers (with standardized cont. values)')
plt.show()


Y2 = zscore(X_cont, ddof=1)

# BOXPLOT: only for standardized continuous values + aligned ################################# 
plt.boxplot(Y2_wo)
plt.xticks(range(1, 6), attributeNames_cont)
plt.title('Boxplot for Standardized Continuous Values - with outliers')
plt.show()

plt.boxplot(Y2)
plt.xticks(range(1, 6), attributeNames_cont)
plt.title('Boxplot for Standardized Continuous Values - without outliers')
plt.show()


'''
From here on we work only with data without outliers:
    - Filtration is based on the quantiles (check main) -
- Data matrix X
    - contains both discrete and continous values, without outliers. 
    attributes: 'age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'oldpeak',
       'slope', 'thal'
- Data matrix X_cont
    - contains only continuous values, without outliers
    attributes: 'age', 'trestbps', 'chol', 'thalach', 'oldpeak'
- Y2: standardized matrix of X_cont

'''

N2, M2 = X_cont.shape

# BOXPLOT based on each class #################################################
# more chance/ less chance of heart attack 
# Also aligned on the same axis for better comparison

plt.figure(figsize=(14,7))
for c in range(C):
    plt.subplot(1, C, c+1)
    class_mask = (y == c)
    plt.boxplot(Y2[class_mask, :])
    plt.title('Boxplot for Class: '+classNames[c])
    plt.xticks(range(1,len(attributeNames_cont)+1), [a[:7] for a in attributeNames_cont], rotation=45)
    y_up = Y2.max()+(Y2.max()-Y2.min())*0.1; y_down = Y2.min()-(Y2.max()-Y2.min())*0.1
    plt.ylim(y_down, y_up)

show()


# SCATTERPLOT for continuous values ############################################

# Scatterplots:
plt.figure(figsize=(12, 10))
for m1 in range(M2):  # loops through attributes - x direction
    for m2 in range(M2):  # loops through attrbutes - y direction
        plt.subplot(M2, M2, m1*M2 + m2 + 1)
        # loops through classes:
        for c in range(C):
            class_mask = (y == c)
            plt.plot(np.array(X_cont[class_mask, m2]), np.array(X_cont[class_mask, m1]), '.', alpha=.5)
      
            # LABELS:
            if m1 == M2-1:  # makes labels when 1st axis
                plt.xlabel(attributeNames_cont[m2])
            else:
                plt.xticks([])
            if m2 == 0:  # makes labels for y axis
                plt.ylabel(attributeNames_cont[m1])
            else:
                plt.yticks([])

plt.legend(classNames)

plt.show()

# CORRELATION MATRIX
corMat = np.corrcoef(X_cont, rowvar=False)
out = pd.DataFrame(corMat, columns=list(attributeNames_cont))
print(out.to_latex(float_format="%.2f"))

# 3D PLOT ###################################################################
# Choice: age, chol and thalach

ind = [0, 1, 2]
colors = ['blue', 'green', 'red']

f = plt.figure()
ax = f.add_subplot(111, projection='3d')
for c in range(C):
    class_mask = (y == c)
    s = ax.scatter(X_cont[class_mask, ind[0]], X_cont[class_mask, ind[1]], X_cont[class_mask, ind[2]], c=colors[c])

ax.view_init(30, 220)
ax.set_xlabel(attributeNames_cont[ind[0]])
ax.set_ylabel(attributeNames_cont[ind[1]])
ax.set_zlabel(attributeNames_cont[ind[2]])

plt.show()

# MATRIX PLOT ###############################################################


plt.figure(figsize=(12,6))
plt.imshow(Y2, interpolation='none', aspect=(8./N), cmap=plt.cm.gray)
plt.xticks(range(5), attributeNames)
plt.xlabel('Attributes')
plt.ylabel('Data objects')
plt.title('Heart Attack Possibility: Data Matrix')
plt.colorbar()

plt.show()
