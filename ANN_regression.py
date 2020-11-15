import numpy as np
from scipy.stats import stats
from sklearn.preprocessing import OneHotEncoder
from auxiliary_functions import rlr_validate, network_validate_regression
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
X = stats.zscore(X)
N, M = X.shape
attributeNames = np.array(['age', 'trestbps', 'chol', 'oldpeak',
                           'female', 'male', 'typical_cp', 'atypical_cp',
                           'no_cp', 'asymptomatic_cp', 'slope_up', 'slope_flat', 'slope_down', 'target0', 'target1'],
                          dtype='<U8')
y = np.asmatrix(y).reshape(N, 1)

K = 10

opt_val_err, opt_n_h_units = network_validate_regression(X, y, np.array([1, 2, 3]))
