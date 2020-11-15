# -*- coding: utf-8 -*-
"""
AUXILIARY FUNCTIONS
"""


import numpy as np
from toolbox_02450.statistics import *
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import train_neural_net
import torch
import concurrent
from CV_split import * #makes sure splitting is only computed once




def network_validate_classification(X,y,h_interval):
    ''' Validate neural network model using 'cvf'-fold cross validation.
        Finds the optimal hidden units from the hidden unit list
        Function returns: optimal test error value, optimal number of hidden units, optimal model after each fold
        
        Parameters:
        X       training data set
        y       vector of values
        h_interval the interval of hidden units. Should start consecutively. Eg: [1,2,3..]  
        
        Returns:
        opt_val_err         validation error for optimum lambda
        opt_n_h_units       optimal number of hidden units
    '''
    
    cvf = 10
    n_replicates = 3
    max_iter = 3000 #this is lower due to computation time
    M = X.shape[1]
    error_rate_matrix = np.empty((cvf,len(h_interval)))

        
    for k, (train_index, test_index) in enumerate(CV1.split(X,y)): 
        print('\nCrossvalidation inner fold: {0}/{1}'.format(k+1,10))    
        
        for h in range(0,len(h_interval)): 
            print("NO OF HIDDEN UNITS:",h+1)
            model = lambda: torch.nn.Sequential(
                                torch.nn.Linear(M, h+1), #M features to H hiden units - h+1 since h starts with 0
                                torch.nn.Tanh(),   # 1st transfer function,
                                torch.nn.Linear(h+1, 1), # H hidden units to 1 output neuron
                                torch.nn.Sigmoid() # final tranfer function
                                )
            loss_fn = torch.nn.BCELoss()
            # Extract training and test set for current CV fold, convert to tensors
            X_train = torch.Tensor(X[train_index,:])
            y_train = torch.Tensor(y[train_index])
            X_test = torch.Tensor(X[test_index,:])
            y_test = torch.Tensor(y[test_index])
            
            # Train the net on training data
            net, final_loss, learning_curve = train_neural_net(model,
                                                               loss_fn,
                                                               X=X_train,
                                                               y=y_train,
                                                               n_replicates=n_replicates,
                                                               max_iter=max_iter)

            
            # Determine estimated class labels for test set
            #y predicted
            y_sigmoid = net(X_test)
            y_test_est = (y_sigmoid>.5).type(dtype=torch.uint8) #set tershold to classify as 0 or 1
        
            # Determine errors and errors
            y_test = y_test.type(dtype=torch.uint8)
        
            e = y_test_est != y_test
            error_rate = (sum(e).type(torch.float)/len(y_test)).data.numpy()
            error_rate_matrix[k,h] = error_rate

        
            print('\n\tBest loss for h= {}: {}\n'.format(h,final_loss))
        
    opt_index = np.argmin(np.mean(error_rate_matrix,axis=0))
    opt_val_err = np.min(np.mean( error_rate_matrix,axis=0))
    opt_n_h_units = h_interval[opt_index]
    print("errors are", np.mean( error_rate_matrix,axis=0))
    print("hidden unit index is",np.argmin(np.mean(error_rate_matrix,axis=0)))
    return opt_val_err, opt_n_h_units
        
    
def rgr_validate_per_fold(train_index, test_index, X, y, lambdas, train_error, test_error, f):
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

        mdl = LogisticRegression(penalty='l2', C=1/lambdas[l], max_iter=10000 )
        
        mdl.fit(X_train, y_train)
        
        y_train_est = mdl.predict(X_train).T
        y_test_est = mdl.predict(X_test).T
        
        train_error[f,l] = np.sum(y_train_est != y_train) / len(y_train)
        test_error[f,l] = np.sum(y_test_est != y_test) / len(y_test)
        
    return train_error, test_error


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
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for train_index, test_index in CV1.split(X,y):
            future = executor.submit(rgr_validate_per_fold, train_index, test_index, X, y, lambdas, train_error, test_error, f)
            train_error, test_error = future.result()
            f=f+1
            print('Inner cross validation fold {0}/{1}...'.format(f,cvf))

    opt_val_err = np.min(np.mean(test_error,axis=0))

    opt_lambda = lambdas[np.argmin(np.mean(test_error,axis=0))]
    train_err_vs_lambda = np.mean(train_error,axis=0)
    test_err_vs_lambda = np.mean(test_error,axis=0)
    
    return opt_val_err, opt_lambda, train_err_vs_lambda, test_err_vs_lambda


def rlr_validate(X, y, lambdas, cvf=10):
    ''' Validate regularized linear regression model using 'cvf'-fold cross validation.
        Find the optimal lambda (minimizing validation error) from 'lambdas' list.
        The loss function computed as mean squared error on validation set (MSE).
        Function returns: MSE averaged over 'cvf' folds, optimal value of lambda,
        average weight values for all lambdas, MSE train&validation errors for all lambdas.
        The cross validation splits are standardized based on the mean and standard
        deviation of the training set when estimating the regularization strength.

        Parameters:
        X       training data set
        y       vector of values
        lambdas vector of lambda values to be validated
        cvf     number of crossvalidation folds

        Returns:
        opt_val_err         validation error for optimum lambda
        opt_lambda          value of optimal lambda
        mean_w_vs_lambda    weights as function of lambda (matrix)
        train_err_vs_lambda train error as function of lambda (vector)
        test_err_vs_lambda  test error as function of lambda (vector)
    '''
    M = X.shape[1]
    w = np.empty((M, cvf, len(lambdas)))
    train_error = np.empty((cvf, len(lambdas)))
    test_error = np.empty((cvf, len(lambdas)))
    f = 0
    y = y.squeeze()
    for train_index, test_index in CV3.split(X, y):
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
        Xty = X_train.T @ y_train
        XtX = X_train.T @ X_train
        for l in range(0, len(lambdas)):
            # Compute parameters for current value of lambda and current CV fold
            # note: "linalg.lstsq(a,b)" is substitue for Matlab's left division operator "\"
            lambdaI = lambdas[l] * np.eye(M)
            lambdaI[0, 0] = 0  # remove bias regularization
            w[:, f, l] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
            # Evaluate training and test performance
            train_error[f, l] = np.power(y_train - X_train @ w[:, f, l].T, 2).mean(axis=0)
            test_error[f, l] = np.power(y_test - X_test @ w[:, f, l].T, 2).mean(axis=0)

        f = f + 1
        print("fold {}/{}".format(f, cvf))

    opt_val_err = np.min(np.mean(test_error, axis=0))
    opt_lambda = lambdas[np.argmin(np.mean(test_error, axis=0))]
    train_err_vs_lambda = np.mean(train_error, axis=0)
    test_err_vs_lambda = np.mean(test_error, axis=0)
    mean_w_vs_lambda = np.squeeze(np.mean(w, axis=1))

    return opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda
