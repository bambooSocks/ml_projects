# -*- coding: utf-8 -*-
"""
AUXILIARY FUNCTIONS
"""


import sklearn.metrics.cluster as cluster_metrics
import numpy as np
import matplotlib.pyplot as plt
from sklearn import model_selection, linear_model
from matplotlib.pyplot import contourf
from matplotlib import cm
from toolbox_02450.statistics import *
from sklearn.linear_model import LogisticRegression
import math
from CV_split import * #makes sure splitting is only computed once

def rgr_validate(X,y,lambdas):
    ''' Validate regularized logistic regression model using 'cvf'-fold cross validation.
        Find the optimal lambda (minimizing validation error) from 'lambdas' list.
        The loss function computed as sum of mismatches divided by N (length of y).
        Function returns: Mismatch Error averaged over 'cvf' folds, optimal value of lambda,
        Mismatch Error train&validation errors for all lambdas.
        The cross validation splits are standardized based on the mean and standard
        deviation of the training set when estimating the regularization strength.
        
        Parameters:
        X       training data set
        y       vector of values
        lambdas vector of lambda values to be validated  
        
        Returns:
        opt_val_err         validation error for optimum lambda
        opt_lambda          value of optimal lambda
        train_err_vs_lambda train error as function of lambda (vector)
        test_err_vs_lambda  test error as function of lambda (vector)
    '''
    cvf = 10
    M = X.shape[1]
    train_error = np.empty((cvf,len(lambdas)))
    test_error = np.empty((cvf,len(lambdas)))
    f = 0
    y = y.squeeze()
    for train_index, test_index in CV1.split(X,y):
        X_train = X[train_index]
        y_train = y[train_index]
        X_test = X[test_index]
        y_test = y[test_index]
        
        # Standardize the training and set set based on training set moments
        mu = np.mean(X_train[:, 1:], 0)
        sigma = np.std(X_train[:, 1:], 0)
        
        X_train[:, 1:] = (X_train[:, 1:] - mu) / sigma
        X_test[:, 1:] = (X_test[:, 1:] - mu) / sigma
        
        # precompute terms

        for l in range(0,len(lambdas)):

            mdl = LogisticRegression(penalty='l2', C=1/lambdas[l] )
            
            mdl.fit(X_train, y_train)
            
            y_train_est = mdl.predict(X_train).T
            y_test_est = mdl.predict(X_test).T
            
            train_error[f,l] = np.sum(y_train_est != y_train) / len(y_train)
            test_error[f,l] = np.sum(y_test_est != y_test) / len(y_test)
    
        f=f+1

    opt_val_err = np.min(np.mean(test_error,axis=0))

    opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
    train_err_vs_lambda = np.mean(train_error,axis=0)
    test_err_vs_lambda = np.mean(test_error,axis=0)
    
    return opt_val_err, opt_lambda, train_err_vs_lambda, test_err_vs_lambda 