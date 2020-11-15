import numpy as np
from toolbox_02450 import jeffrey_interval
from toolbox_02450 import mcnemar

from regression_b import *
from ANN_regression import *

'''
SETUP I

Returning to the post-surgical example, the conclusions we might arrive at under setup I therefore
have to be stated conditional on the dataset D we have only. We might (for instance) find model MA is significantly better
than MB, but our conclusion can only be said to have been tested (and therefore, be valid) in the
case the models are trained on D.
'''

y_true = np.squeeze(np.asarray(y.reshape(1,282)))

y_est_A = best_regression_model(np.concatenate((np.ones((X.shape[0],1)),X),1))
y_est_B = best_ANN_model(torch.Tensor(X)).detach().numpy().reshape(X.shape[0])
y_est_C = y.mean() + np.zeros(y.shape[0])


yhat = np.column_stack((y_est_A, y_est_B, y_est_C))



print("\nCOMPARISON BETWEEN REGULARIZED LINEAR REGRESSION AND NEURAL NETWORK MODEL")


'''
Comparison between Regularized Regularized Logistic Regression Model and ANN Model:

'''
#Difference in performance between Linear Regression Model and Base Model: use McNemar test
print("\nCOMPARISON BETWEEN REGULARIZED LINEAR REGRESSION AND BASE MODEL")


'''
Comparison between Regularized Linear Regression Model and Base Model:
    
'''

#Difference in performance between ANN Model and Base Model: use McNemar test
print("\nCOMPARISON BETWEEN NEURAL NETWORK AND BASE MODEL")



'''
Comparison between ANN Model and Base Model:
    
'''

print("REGULARIZED LINEAR REGRESSION")
print("Testing errors for the LR are", np.round(Error_test_rlr,2))
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


