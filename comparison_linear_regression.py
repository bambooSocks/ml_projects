from matplotlib.pyplot import figure, plot, xlabel, ylabel, show
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn import model_selection
import sklearn.tree
import scipy.stats
import numpy as np, scipy.stats as st

# requires data from exercise 5.1.5
from ex5_1_5 import *

X,y = X[:, :10], X[:, 10:]
# This script crates predictions from three KNN classifiers using cross-validation

K = 16
CV = model_selection.KFold(n_splits=K, shuffle = True, random_state=0) # seed seed value to 0 for same selection

zA = []
zB = []

for train_index, test_index in CV.split(X):
    
    # extract training and test set for current CV fold
    X_train = X[train_index,:]
    y_train = y[train_index]
    X_test = X[test_index,:]
    y_test = y[test_index]

    # Fit the models
    mA = sklearn.linear_model.LinearRegression().fit(X_train,y_train)
    mB = sklearn.tree.DecisionTreeRegressor().fit(X_train, y_train)

    y_est_A = mA.predict(X_test)
    y_est_B = mB.predict(X_test)[:,np.newaxis]  #  justsklearnthings

    zA_temp = np.abs(y_test - y_est_A ) ** 2 
    zB_temp = np.abs(y_test - y_est_B ) ** 2
    
    zA.append(zA_temp)
    zB.append(zB_temp)
    

# From the list that contains all the error vectors for each K-fold, create an array of mean errors acrross the K-folds       
zA = np.stack(zA, axis=1) 
zB = np.stack(zB, axis=1)

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