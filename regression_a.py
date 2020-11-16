import numpy as np
from scipy.stats import stats
from sklearn.preprocessing import OneHotEncoder
from auxiliary_functions import rlr_validate
import matplotlib.pyplot as plt

from data_aquisition import *

enc = OneHotEncoder()
# we get our discrete variables: sex, cp, slope and target
X_discrete = np.stack((X_sel[:, 1], X_sel[:, 2], X_sel[:, 7], y), axis=-1)
enc.fit(X_discrete)
X_enc = enc.transform(X_discrete).toarray()
# stack on top of the continuous ones
y = X_cont[:,3].T
y = stats.zscore(y)
y_label = "thalach"
X_cont = np.delete(X_cont, 3, axis=1)
X = np.column_stack((X_cont, X_enc))
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
N, M = X.shape
attributeNames = np.array(['Offset', 'age', 'trestbps', 'chol', 'oldpeak',
                           'female', 'male', 'typical_cp', 'atypical_cp',
                           'no_cp', 'asymptomatic_cp', 'slope_up', 'slope_flat', 'slope_down', 'target0', 'target1'],
                          dtype='<U8')

K = 10

# Values of lambda
lambdas = np.logspace(0, 5, 50)

opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X, y, lambdas, cvf=K)

plt.figure(figsize=(8, 8))

plt.title('Model error vs regularization factor (lambda)')
plt.semilogx(opt_lambda, opt_val_err, color='cyan', markersize=12, marker='o')
plt.loglog(lambdas, train_err_vs_lambda.T, 'b-', lambdas, test_err_vs_lambda.T, 'r-')
plt.xlabel('Regularization factor')
plt.ylabel('Error rate')
plt.legend(['Test minimum', 'Training error', 'Validation error'])
plt.grid()
plt.show()

print('Regularized linear regression:')
print('- Training error: {0}'.format(train_err_vs_lambda.mean()))
print('- Test error:     {0}'.format(test_err_vs_lambda.mean()))

print(opt_lambda)
