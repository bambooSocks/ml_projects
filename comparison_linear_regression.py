from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import sklearn.tree
import scipy.stats as st

from regression_b import *
from ANN_regression import *


K = 10
CV = model_selection.KFold(n_splits=K, shuffle = True, random_state=0) # seed seed value to 0 for same selection

zA = []
zB = []

y_est_A = best_regression_model(np.concatenate((np.ones((X.shape[0],1)),X),1))
y_est_B = best_ANN_model(torch.Tensor(X)).detach().numpy().reshape(X.shape[0])

zA = np.abs(y_test - y_est_A) ** 2
zB = np.abs(y_test - y_est_B) ** 2


# Code below is used to create a usable matrix from the stacked error vectors for each K-fold
# Matrix is defined as: Number of calculated errors (size of the training set) x Kfold
# np.sort and np.shape are used to trim the dyA and dyB matrices, as np.stack creates (train, Kfold, 1) numpy arrays
# and last 1-dimensional shape is pointless.
zA = np.reshape(zA, (np.sort(np.shape(zA))[-1], np.sort(np.shape(zA))[-2]))
zB = np.reshape(zB, (np.sort(np.shape(zB))[-1], np.sort(np.shape(zB))[-2]))

# Compute the mean of all errors across all the K-folds
zA  = np.mean(zA, axis = 1)
zB  = np.mean(zB, axis = 1)
    
alpha = 0.05
# compute confidence interval of model A
CIA = st.t.interval(1-alpha, df=len(zA)-1, loc=np.mean(zA), scale=st.sem(zA))  # Confidence interval

# compute confidence interval of model B
CIB = st.t.interval(1-alpha, df=len(zB)-1, loc=np.mean(zB), scale=st.sem(zB))

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
z = zA - zB
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value

print("CIA:", CIA)
print("CIB:", CIB)
print("Mean estimation (thetaHat):", np.mean(z))
print("CI:", CI)
print("p-value:", p)

'''
The CI and p value is always changing, since test set and traning set is randomly split. 
'''