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
y_torch_B = np.squeeze(best_ANN_model(torch.Tensor(X))>.5)
y_est_B = y_torch_B.detach().numpy().reshape(X.shape[0]) #set tershold to classify as 0 or 1
y_est_C = np.bincount(y_true).argmax() + np.zeros(y.shape[0])

#y_est_A = best_regression_model(np.concatenate((np.ones((X.shape[0],1)),X),1))
#y_est_B = best_ANN_model(torch.Tensor(X)).detach().numpy().reshape(X.shape[0])
#y_est_C = y.mean() + np.zeros(y.shape[0])


yhat = np.column_stack((y_est_A, y_est_B, y_est_C))


#Difference in performance between LOGISTIC Regression Model and ANN Model: use McNemar test
# Compute the Jeffreys interval
alpha = 0.05



print("\nCOMPARISON BETWEEN REGULARIZED LOGISTIC REGRESSION AND NEURAL NETWORK MODEL")
[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,1], alpha=alpha)
print("\ntheta = theta_LR-theta_ANN point estimate", thetahat, " CI: ", CI, "p-value", p)

'''
Comparison between Regularized Regularized Logistic Regression Model and ANN Model:

OUTPUT:
Result of McNemars test using alpha= 0.05
Comparison matrix n
[[194.  17.]
 [ 39.  32.]]
Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] =  (-0.12902248070670852, -0.026803468355817817)
p-value for two-sided test A and B have same accuracy (exact binomial test): p= 0.004561532746408221

theta = theta_LR-theta_ANN point estimate -0.07801418439716312  CI:  (-0.12902248070670852, -0.026803468355817817) p-value 0.004561532746408221

-- INTERPRETATION -- 

   
CI does not contain 0 and it is negative. 
One could say that the Logistic Regression Model performs better than the Neural Network Model.
The low p-value (p < 0.01) indicates this result is not by chance. 

'''
#Difference in performance between LOGISTIC Regression Model and Base Model: use McNemar test
print("\nCOMPARISON BETWEEN REGULARIZED LOGISTIC REGRESSION AND BASE MODEL")

[thetahat, CI, p] = mcnemar(y_true, yhat[:,0], yhat[:,2], alpha=alpha)
print("\ntheta = theta_LR-theta_Base_Model point estimate", thetahat, " CI: ", CI, "p-value", p)

'''
Comparison between Regularized Logistic Regression Model and Base Model:

OUTPUT: 
Result of McNemars test using alpha= 0.05
Comparison matrix n
[[151.  60.]
 [  7.  64.]]
Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] =  (0.13528977757692506, 0.24006879043655105)
p-value for two-sided test A and B have same accuracy (exact binomial test): p= 1.3280368233066499e-11

theta = theta_LR-theta_Base_Model point estimate 0.1879432624113475  CI:  (0.13528977757692506, 0.24006879043655105) p-value 1.3280368233066499e-11

-- INTERPRETATION --   
CI is positive so this means that there is a clear difference in the performance between the two models.
The Logistic Regression Model performs better than the Baseline Model (which is expected). 
The very low p value (p<0.01) indicates that this result is not by chance.  

'''

#Difference in performance between ANN Model and Base Model: use McNemar test
print("\nCOMPARISON BETWEEN NEURAL NETWORK AND BASE MODEL")

[thetahat, CI, p] = mcnemar(y_true, yhat[:,1], yhat[:,2], alpha=alpha)
print("\ntheta = theta_ANN-theta_Base_Model point estimate", thetahat, " CI: ", CI, "p-value", p)

'''
Comparison between ANN Model and Base Model:
    
OUTPUT: 
Result of McNemars test using alpha= 0.05
Comparison matrix n
[[134.  99.]
 [ 24.  25.]]
Approximate 1-alpha confidence interval of theta: [thetaL,thetaU] =  (0.19484910697897528, 0.33566360845707166)
p-value for two-sided test A and B have same accuracy (exact binomial test): p= 5.1659499319135495e-12

theta = theta_ANN-theta_Base_Model point estimate 0.26595744680851063  CI:  (0.19484910697897528, 0.33566360845707166) p-value 5.1659499319135495e-12

-- INTERPRETATION --
CI is positive, so this means that the ANN Model performs better than the baseline model. 
This is also confirmed by the low p-value.     
    
'''

print("REGULARIZED LOGISTIC REGRESSION")
print("Testing errors for the LogR are \n", np.round(Error_test_rlr*100,2))
print("With optimal regularizing parameters: \n", np.round(lambdas_vect,2))
print("LR: Minimum test error: " + str(np.round(min_error*100,2)) + ' % with lambda ' + str(best_lambda[0]))

print()

print("NEURAL NETWORK MODEL")
print("Error rates for ANN model are: \n",np.round(errors*100,2))
print("Optimal hidden units for each fold: \n",h_optimal_list)
print("Optimal test error is {}% with no of hidden units {}".format(np.round(opt_val_err*100,2),opt_n_h_units))

print()

print("BASE MODEL")
print("Testing errors for the Base Model are \n", np.round(Error_test_nofeatures,2))

'''
Conclusion: Overall both Logistic Regression Model and the Artificial Neural Network Model perform better than the baseline model.
This means that those classifiers are quite efficient at predicting the target value, Log Regression predicts the target right about 92% of the time
and Neural Networl Model predicts the target right about 80% of the time.   
Performance is quite good for both of them, but when it comes to selecting the best model, one should choose the Regularized Logistic Regression model.
Since we are doing the comparison using Setup I, SETUP I the conclusions we might arrive at is only conditional to this particular dataset. 
                                           
'''

