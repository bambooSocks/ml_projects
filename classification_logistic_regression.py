import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from auxiliary_functions import rgr_validate

from data_aquisition import *

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

lambdas = np.logspace(-2, 2, 50)

opt_val_err, opt_lambda, train_err_vs_lambda, test_err_vs_lambda = rgr_validate(X, y, lambdas)

opt_lambda_2 = np.round(opt_lambda,2)

plt.figure(figsize=(8,8))
plt.title('Classification errors plotted against lambda interval')        
plt.semilogx(opt_lambda, opt_val_err, color='cyan', markersize=12, marker='o')
plt.text(1e-2, 1.9e-1, "Minimum test error: " + str(round(opt_val_err*100,2)) + ' % \nat optimal lambda: {}'.format(opt_lambda_2))
plt.loglog(lambdas,train_err_vs_lambda.T,'b-',lambdas,test_err_vs_lambda.T,'r-')
plt.xlabel('Regularization factor')
plt.ylabel('Error rate - last inner fold')
plt.legend(['Test minimum','Training error','Validation error'])
plt.grid() 

#complexity of model decreases as we increase reg. strength lambda

print('Best classification error is {}% with best regularization term: {}'.format(round(opt_val_err*100,2),opt_lambda_2)) 
'''
Note: 
As one can see, the minimum test error is 26.49% which is quite high.
This is due to the fact that we have used all the attributes and only the regularization term for controlling model complexity. 
We have tried cutting off the attributes based on their importance (have used the REF package). However the training error is way worse. 
This means our model will classify y correctly about 1/4 times. 

'''