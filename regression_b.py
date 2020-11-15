import numpy as np
from sklearn.preprocessing import OneHotEncoder
from auxiliary_functions import rlr_validate
from CV_split import CV4
import matplotlib.pyplot as plt

from data_aquisition import *

enc = OneHotEncoder()
# we get our discrete variables: sex, cp, slope and target
X_discrete = np.stack((X_sel[:, 1], X_sel[:, 2], X_sel[:, 7], y), axis=-1)
enc.fit(X_discrete)
X_enc = enc.transform(X_discrete).toarray()
# stack on top of the continuous ones
y = X_cont[:,2].T
y_label = "chol"
X_cont = np.delete(X_cont, 2, axis=1)
X = np.column_stack((X_cont, X_enc))
# Add offset attribute
X = np.concatenate((np.ones((X.shape[0],1)),X),1)
N, M = X.shape
attributeNames = np.array(['Offset', 'age', 'trestbps', 'thalach', 'oldpeak',
                           'female', 'male', 'typical_cp', 'atypical_cp',
                           'no_cp', 'asymptomatic_cp', 'slope_up', 'slope_flat', 'slope_down', 'target0', 'target1'],
                          dtype='<U8')

K = 10

# Values of lambda
lambdas = np.logspace(-2, 10, 50)

Error_train = np.empty((K, 1))
Error_test = np.empty((K, 1))
Error_train_rlr = np.empty((K, 1))
Error_test_rlr = np.empty((K, 1))
w_rlr = np.empty((M, K))
mu = np.empty((K, M - 1))
sigma = np.empty((K, M - 1))
w_noreg = np.empty((M, K))

k = 0
for train_index, test_index in CV4.split(X, y):

    # extract training and test set for current CV fold
    X_train = X[train_index]
    y_train = y[train_index]
    X_test = X[test_index]
    y_test = y[test_index]
    internal_cross_validation = 10

    opt_val_err, opt_lambda, mean_w_vs_lambda, train_err_vs_lambda, test_err_vs_lambda = rlr_validate(X_train, y_train,
                                                                                                      lambdas,
                                                                                                      internal_cross_validation)

    # Standardize outer fold based on training set, and save the mean and standard
    # deviations since they're part of the model (they would be needed for
    # making new predictions) - for brevity we won't always store these in the scripts
    mu[k, :] = np.mean(X_train[:, 1:], 0)
    sigma[k, :] = np.std(X_train[:, 1:], 0)

    X_train[:, 1:] = (X_train[:, 1:] - mu[k, :]) / sigma[k, :]
    X_test[:, 1:] = (X_test[:, 1:] - mu[k, :]) / sigma[k, :]

    Xty = X_train.T @ y_train
    XtX = X_train.T @ X_train

    # Estimate weights for the optimal value of lambda, on entire training set
    lambdaI = opt_lambda * np.eye(M)
    lambdaI[0, 0] = 0  # Do no regularize the bias term
    w_rlr[:, k] = np.linalg.solve(XtX + lambdaI, Xty).squeeze()
    # Compute mean squared error with regularization with optimal lambda
    Error_train_rlr[k] = np.square(y_train - X_train @ w_rlr[:, k]).sum(axis=0) / y_train.shape[0]
    Error_test_rlr[k] = np.square(y_test - X_test @ w_rlr[:, k]).sum(axis=0) / y_test.shape[0]

    # Display the results for the last cross-validation fold
    if k == K - 1:
        plt.figure(k, figsize=(12, 8))
        plt.subplot(1, 2, 1)
        plt.semilogx(lambdas, mean_w_vs_lambda.T[:, 1:], '.-')  # Don't plot the bias term
        plt.xlabel('Regularization factor')
        plt.ylabel('Mean Coefficient Values')
        plt.grid()
        # You can choose to display the legend, but it's omitted for a cleaner
        # plot, since there are many attributes
        # legend(attributeNames[1:], loc='best')

        plt.subplot(1, 2, 2)
        plt.title('Optimal lambda: 1e{0}'.format(np.log10(opt_lambda)))
        plt.loglog(lambdas, train_err_vs_lambda.T, 'b.-', lambdas, test_err_vs_lambda.T, 'r.-')
        plt.semilogx(opt_lambda, opt_val_err, color='cyan', markersize=12, marker='o')
        plt.xlabel('Regularization factor')
        plt.ylabel('Squared error (crossvalidation)')
        plt.legend(['Train error', 'Validation error'])
        plt.grid()

    k += 1

plt.show()
# Display results
print('Linear regression without feature selection:')
print('- Training error: {0}'.format(Error_train.mean()))
print('- Test error:     {0}'.format(Error_test.mean()))
print('Regularized linear regression:')
print('- Training error: {0}'.format(Error_train_rlr.mean()))
print('- Test error:     {0}'.format(Error_test_rlr.mean()))
