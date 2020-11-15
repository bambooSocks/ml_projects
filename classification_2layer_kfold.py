# -*- coding: utf-8 -*-
"""
Created on Fri Nov 13 15:27:51 2020

@author: bejen
"""
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import OneHotEncoder
from matplotlib.pylab import (figure, semilogx, loglog, xlabel, ylabel, legend, 
                           title, subplot, show, grid)


from data_aquisition import * 
from auxiliary_functions import rgr_validate #does the inner cross-validation for regularized logistic regression, with cvf=10 - see documentation
from CV_split import * #uses same computed stratification splits for cross validation

## Data Prepping ###########################################################################
enc = OneHotEncoder()
# we get our discrete variables: sex, cp, slope
X_discrete = np.stack((X_sel[:,1],X_sel[:,2],X_sel[:,7]),axis=-1)
enc.fit(X_discrete)
X_enc = enc.transform(X_discrete).toarray()
# stack on top of the continuous ones
X = np.column_stack((X_sel[:,0],X_sel[:,3:7],X_enc))
#update attributeNames
attributeNames = np.array(['age','trestbps', 'chol', 'thalach', 'oldpeak',
       'female','male','typical_cp','atypical_cp',
       'no_cp','asymptomatic_cp','slope_up','slope_flat','slope_down'], dtype='<U8')
N, M = X.shape

##############################################################################################

## Crossvalidation
# Create crossvalidation partition for evaluation
K = 10

# Values of lambda
lambdas = np.logspace(-2, 2, 50)

lambdas_vect = np.empty((K,1))
Error_train_rlr = np.empty((K,1))
Error_test_rlr = np.empty((K,1))
Error_train_nofeatures = np.empty((K,1))
Error_test_nofeatures = np.empty((K,1))
mu = np.empty((K, M-1))
sigma = np.empty((K, M-1))
s=X.shape[1] +1
coefficient_matrix = np.zeros((K,s))
models_lr = [] #list to store the best model after each outer cv-fold

k=0
for train_index, test_index in CV2.split(X,y):
    
    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    
    print('Please wait: Outer cross validation fold {0}/{1}...'.format(k+1,K))
    #rgr_validate does the inner cross-validation with cvf=10
    opt_val_err, opt_lambda, train_err_vs_lambda, test_err_vs_lambda = rgr_validate(X_train, y_train, lambdas)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)
    
    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :] ) / sigma[k, :] 
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :] ) / sigma[k, :]

    # Compute mean squared error without using the input data at all
    Error_train_nofeatures[k] = np.sum(y_train != np.bincount(y_train).argmax())/y_train.shape[0]
    Error_test_nofeatures[k] = np.sum(y_test != np.bincount(y_test).argmax())/y_test.shape[0]

    #@ignore_warnings(category = ConvergenceWarning)
    mdl2 = LogisticRegression(penalty='l2', C=1/opt_lambda, max_iter=10000 )
    models_lr.append(mdl2)
    
    mdl2.fit(X_train, y_train)

    y_tr_est = mdl2.predict(X_train).T
    y_tst_est = mdl2.predict(X_test).T
    
    
    Error_train_rlr[k] = np.sum(y_train != y_tr_est)/y_train.shape[0]
    Error_test_rlr[k] = np.sum(y_test != y_tst_est)/y_test.shape[0]
    lambdas_vect[k] = opt_lambda

    # magnitude of the coefficients - add in coeff matrix
    w_est = mdl2.coef_[0] 
    coef_vector = np.concatenate((mdl2.intercept_,np.squeeze(mdl2.coef_)))
    coefficient_matrix[k,:] = coef_vector
        
   
    # Display the results for the last cross-validation fold
    if k == 5: #we take the fold according to the found out optimal index (opt_idx)
        plt.figure(figsize=(8,8))
        #plt.figure(k, figsize=(12,8))
        title('Classification error for best cv-fold with lambda: 1e{0}'.format(np.round(np.log10(opt_lambda),2)))        
        plt.semilogx(opt_lambda, opt_val_err, color='cyan', markersize=12, marker='o')
        plt.text(1e-2, 2.0e-1, "Minimum test error: " + str(round(opt_val_err*100,2)) + ' % at optimal lambda: 1e{0}'.format(np.round(np.log10(opt_lambda),2)))
        loglog(lambdas,train_err_vs_lambda.T,'b-',lambdas,test_err_vs_lambda.T,'r-')
        xlabel('Regularization factor')
        ylabel('Error rate - last inner fold')
        legend(['Test minimum','Training error','Validation error'])
        grid()    
    
    k+=1
    
min_error = np.min(Error_test_rlr)
opt_idx = np.argmin(Error_test_rlr)
best_LR_model = models_lr[opt_idx]
# LogisticRegression(C=0.44984326689694437)
best_lambda = lambdas_vect[opt_idx]


print("Training errors for LogReg are", np.round(Error_train_rlr,2))
print("Testing errors for LogReg are", np.round(Error_test_rlr,2))
print("Lambdas are", lambdas_vect)
print("Weights for the optimal model are: \n", coefficient_matrix[opt_idx,:])
array_best_lambda = np.round(np.log10(best_lambda),2)
print("Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(array_best_lambda[0]))

print()

print("Testing errors for the Base Model are", np.round(Error_test_nofeatures,2))

print()
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
print('- R^2 train:     {0}'.format((Error_train_nofeatures.sum()-Error_train_rlr.sum())/Error_train_nofeatures.sum()))
print('- R^2 test:     {0}\n'.format((Error_test_nofeatures.sum()-Error_test_rlr.sum())/Error_test_nofeatures.sum()))

