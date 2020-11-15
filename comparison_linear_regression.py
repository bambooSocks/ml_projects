import numpy as np
from toolbox_02450 import jeffrey_interval
from toolbox_02450 import mcnemar

from regression_b import *
from ANN_regression import *

'''
SETUP I

Returning to the post-surgical example, the conclusions we might arrive at under setup I therefore
have to be stated conditional on D. We might (for instance) find model MA is significantly better
than MB, but our conclusion can only be said to have been tested (and therefore, be valid) in the
case the models are trained on D.
'''

y_est_A = best_regression_model(np.concatenate((np.ones((X.shape[0],1)),X),1))
y_est_B = best_ANN_model(torch.Tensor(X)).detach().numpy().reshape(X.shape[0])
y_est_C = y.mean() + np.zeros(y.shape[0])

y_true = np.squeeze(np.asarray(y.reshape(1,282)))
yhat = np.column_stack((y_est_A, y_est_B, y_est_C))

#for LR model choose: yhat[:,0]
# Compute the Jeffreys interval
alpha = 0.05
[thetahatA, CIA] = jeffrey_interval(y_true, yhat[:,0], alpha=alpha)
print("Theta point estimate for Linear Regression Model", thetahatA, " CI: ", CIA)

# for ANN Model choose: yhat[:,1]
[thetahatB, CIB] = jeffrey_interval(y_true, yhat[:,1], alpha=alpha)

print("\nTheta point estimate for ANN Model", thetahatB, " CI: ", CIB)

# for Base Model choose: yhat[:,2]
[thetahatC, CIC] = jeffrey_interval(y_true, yhat[:,2], alpha=alpha)

print("\nTheta point estimate for Base Model", thetahatC, " CI: ", CIC)



#Difference in performance between Linear Regression Model and ANN Model: use McNemar test
# Compute the Jeffreys interval
alpha = 0.05
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,1], alpha=alpha)

print("\ntheta = theta_LR-theta_ANN point estimate", thetahat, " CI: ", CI, "p-value", p)

'''
Comparison between Regularized Linear Regression Model and ANN Model:
    
CI interval barely doesn’t contain 0, which is weak
evidence towards MB having a relatively higher accuracy than MA. Meanwhile,
the p-value is relatively high, indicating the result is likely due to chance. All in
all the result is inconclusive and we should not conclude MB is better than MA.
'''
#Difference in performance between Linear Regression Model and Base Model: use McNemar test

[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,2], alpha=alpha)
print("\ntheta = theta_LR-theta_Base_Model point estimate", thetahat, " CI: ", CI, "p-value", p)

'''
Comparison between Regularized Linear Regression Model and Base Model:
    
CI doesnøt contain 0, so one can say MC performs slightly better than MA.
There is a relatively higher difference in performance between
MA and MC. The performance difference
θ is estimated to be between (approximately) 0.05 and 0.1. The confidence
interval is therefore well clear of 0 and the low p-value (p < 0.01) indicates the
result is not likely to be due to chance.
'''

#Difference in performance between ANN Model and Base Model: use McNemar test

[thetahat, CI, p] = mcnemar(y_true, yhat[:,1], yhat[:,2], alpha=alpha)
print("\ntheta = theta_ANN-theta_Base_Model point estimate", thetahat, " CI: ", CI, "p-value", p)

'''
Comparison between ANN Model and Base Model:
'''
