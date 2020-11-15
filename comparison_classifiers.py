import numpy as np
from toolbox_02450 import jeffrey_interval
from toolbox_02450 import mcnemar

from classification_2layer_kfold import *
from ANN_2level_classifier import *

'''
SETUP I

Returning to the post-surgical example, the conclusions we might arrive at under setup I therefore
have to be stated conditional on D. We might (for instance) find model MA is significantly better
than MB, but our conclusion can only be said to have been tested (and therefore, be valid) in the
case the models are trained on D.
'''

y_true = np.squeeze(np.asarray(y.reshape(1,282)))
y_est_A = best_LR_model.predict(X)
y_torch_B = best_ANN_model(torch.Tensor(X))>.5
y_est_B = y_torch_B.detach().numpy().reshape(X.shape[0]) #set tershold to classify as 0 or 1
y_est_C = np.bincount(y_true).argmax() + np.zeros(y.shape[0])

#y_est_A = best_regression_model(np.concatenate((np.ones((X.shape[0],1)),X),1))
#y_est_B = best_ANN_model(torch.Tensor(X)).detach().numpy().reshape(X.shape[0])
#y_est_C = y.mean() + np.zeros(y.shape[0])


yhat = np.column_stack((y_est_A, y_est_B, y_est_C))


#Difference in performance between LOGISTIC Regression Model and ANN Model: use McNemar test
# Compute the Jeffreys interval
alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,1], alpha=alpha)

print("\nCOMPARISON BETWEEN REGULARIZED LOGISTIC REGRESSION AND NEURAL NETWORK MODEL")
print("\ntheta = theta_LR-theta_ANN point estimate", thetahat, " CI: ", CI, "p-value", p)

'''
Comparison between Regularized Regularized Logistic Regression Model and ANN Model:


-- EXAMPLE: WHAT TO WRITE AS INTERPRETATION --    
CI doesnøt contain 0, so one can say MC performs slightly better than MA.
There is a relatively higher difference in performance between
MA and MC. The performance difference
θ is estimated to be between (approximately) 0.05 and 0.1. The confidence
interval is therefore well clear of 0 and the low p-value (p < 0.01) indicates the
result is not likely to be due to chance.

'''
#Difference in performance between LOGISTIC Regression Model and Base Model: use McNemar test
print("\nCOMPARISON BETWEEN REGULARIZED LOGISTIC REGRESSION AND BASE MODEL")

[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,2], alpha=alpha)
print("\ntheta = theta_LR-theta_Base_Model point estimate", thetahat, " CI: ", CI, "p-value", p)

'''
Comparison between Regularized Logistic Regression Model and Base Model:

   

'''

#Difference in performance between ANN Model and Base Model: use McNemar test
print("\nCOMPARISON BETWEEN NEURAL NETWORK AND BASE MODEL")

[thetahat, CI, p] = mcnemar(y_true, yhat[:,1], yhat[:,2], alpha=alpha)
print("\ntheta = theta_ANN-theta_Base_Model point estimate", thetahat, " CI: ", CI, "p-value", p)

'''
Comparison between ANN Model and Base Model:
'''

print("REGULARIZED LOGISTIC REGRESSION")
print("Testing errors for the LogR are", np.round(Error_test_rlr,2))
print("With optimal regularizing parameters:", lambdas_vect)
print("LR: Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(array_best_lambda[0]))

print()

print("NEURAL NETWORK MODEL")
print("Error rates for ANN model are: ",errors)
print("Optimal hidden units for each fold: ",h_optimal_list)
print("Optimal test error is {}% with no of hidden units {}".format(np.round(opt_val_err*100,2),opt_n_h_units))

print()

print("BASE MODEL")
print("Testing errors for the Base Model are", np.round(Error_test_nofeatures,2))

