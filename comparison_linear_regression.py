import numpy as np
from toolbox_02450 import jeffrey_interval
from toolbox_02450 import mcnemar
import numpy as np, scipy.stats as st
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

y_est_A = best_regression_model(np.concatenate((np.ones((X.shape[0],1)),stats.zscore(X)),1))
y_est_B = best_ANN_model(torch.Tensor(stats.zscore(X))).detach().numpy().reshape(X.shape[0])
y_est_C = y.mean() + np.zeros(y.shape[0])



yhat = np.column_stack((y_est_A, y_est_B, y_est_C))

zA = np.abs(y_true - yhat[:,0] ) ** 2
zB = np.abs(y_true - yhat[:,1] ) ** 2
zC = np.abs(y_true - yhat[:,2] ) ** 2

alpha = 0.05

print("\nCOMPARISON BETWEEN REGULARIZED LINEAR REGRESSION AND NEURAL NETWORK MODEL")

# Compute confidence interval of z = zA-zB and p-value of Null hypothesis
z = zA - zB
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value

'''
Comparison between Regularized Regularized Logistic Regression Model and ANN Model:

'''
#Difference in performance between Linear Regression Model and Base Model: use McNemar test
print("\nCOMPARISON BETWEEN REGULARIZED LINEAR REGRESSION AND BASE MODEL")

z = zA - zC
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value

'''
Comparison between Regularized Linear Regression Model and Base Model:
    
'''

#Difference in performance between ANN Model and Base Model: use McNemar test
print("\nCOMPARISON BETWEEN NEURAL NETWORK AND BASE MODEL")

z = zB - zC
CI = st.t.interval(1-alpha, len(z)-1, loc=np.mean(z), scale=st.sem(z))  # Confidence interval
p = st.t.cdf( -np.abs( np.mean(z) )/st.sem(z), df=len(z)-1)  # p-value

'''
Comparison between ANN Model and Base Model:
    
'''

print("LINEAR REGRESSION")
print("Testing errors for the LR are \n", np.round(Error_test_rlr,2))
print("With optimal regularizing parameters: \n", lambdas_vect)
print("LR: Minimum test error: " + str(np.round(min_error*100,2)) + ' % with reg parameter' + str(round(best_lambda[0],2)))

print()

print("NEURAL NETWORK MODEL")
print("Test errors for ANN per k fold are: ", np.round((np.asarray(errors_sqrt)),4))
print("With ptimal hidden units for each fold: ", h_optimal_list)
print("Optimal test error is {}% with no of hidden units {}".format(np.round(opt_val_err*100,2),opt_n_h_units))

print()

print("BASE MODEL")
print("Testing errors for the Base Model are \n", np.round(Error_test_nofeatures,2))


