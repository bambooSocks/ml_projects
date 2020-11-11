import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from toolbox_02450 import rocplot, confmatplot
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_selection import RFE

from data_aquisition import *

# Removing thal from dataset (as discussed in previous report)
attributeNames = np.delete(sel_attr, -1, axis=0)
# ['age', 'sex', 'cp', 'trestbps', 'chol', 'thalach', 'oldpeak', 'slope']

X = np.delete(X_sel, -1, axis=1)

# We select use the RFE package to recursively select the features of most importance:


logreg = LogisticRegression(fit_intercept = False)
logreg.fit(X,y)
selector = RFE(logreg, n_features_to_select=1)
selector = selector.fit(X, y)
order = selector.ranking_
print(order)

feature_ranks = []
for i in order:
    feature_ranks.append(f"{i}. {attributeNames[i-1]}")
print("Most important feature ranking which should be considered for a future model is:", feature_ranks)

'''
Most important features ranking: ['5. chol', '3. cp', '1. age', '7. oldpeak', '8. slope', '6. thalach', '2. sex', '4. trestbps']    
However for this project we will use all attributes and use a regularization term to control complexity.
Performance of different models (ANN, CT, KNN, NB) will be analyzed afterwards.
'''
# We treat thalach as categorical. since we do not know which angina pain type is worse (typical or atypical)

# One out of hot encoding:
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


# Crossvalidation partition for evaluation
# using stratification and 95 pct. split between training and test 
K = 10 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.95, stratify=y, random_state=0)

# we have set a seed so results are reproducible (random_state=0)

# Standardize the training and test set based on training set mean and std
mu = np.mean(X_train, 0)
sigma = np.std(X_train, 0)

X_train = (X_train - mu) / sigma
X_test = (X_test - mu) / sigma

lambda_interval = np.logspace(-8, 5, 50)
train_error_rate = np.zeros(len(lambda_interval))
test_error_rate = np.zeros(len(lambda_interval))
coefficient_norm = np.zeros(len(lambda_interval))
coefficient_matrix = np.zeros((len(lambda_interval),15))
for k in range(0, len(lambda_interval)):
    #regularization regression - L2
    mdl = LogisticRegression(penalty='l2', C=1/lambda_interval[k] )
    
    mdl.fit(X_train, y_train)

    y_train_est = mdl.predict(X_train).T
    y_test_est = mdl.predict(X_test).T
    
    train_error_rate[k] = np.sum(y_train_est != y_train) / len(y_train)
    test_error_rate[k] = np.sum(y_test_est != y_test) / len(y_test)

# magnitude of the coefficients
    w_est = mdl.coef_[0] 
    coefficient_norm[k] = np.sqrt(np.sum(w_est**2))
    coef_vector = np.concatenate((mdl.intercept_,np.squeeze(mdl.coef_)))
    coefficient_matrix[k,:] = coef_vector

min_error = np.min(test_error_rate)
opt_lambda_idx = np.argmin(test_error_rate) #26
opt_wieghts = coefficient_matrix[26,:]
#  array([ 0.99453237,  0.72318361,  0.68650974,  0.61346421,  1.58205608,
#       -2.04352359,  1.92511527, -1.92511527, -1.2582287 ,  0.00290383,
#       -0.97304672,  2.50443067, -0.01087359, -0.28967932,  0.29834006])
opt_lambda = lambda_interval[opt_lambda_idx]

# 0.07906043210907701

plt.figure(figsize=(8,8))
plt.semilogx(lambda_interval, train_error_rate*100)
plt.semilogx(lambda_interval, test_error_rate*100)
plt.semilogx(opt_lambda, min_error*100, 'o')
plt.text(1e-8, 5, "Minimum test error: " + str(np.round(min_error*100,2)) + ' % at 1e' + str(np.round(np.log10(opt_lambda),2)))
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.ylabel('Error rate (%)')
plt.title('Classification error')
plt.legend(['Training error','Test error','Test minimum'],loc='upper right')
plt.ylim([0, 50])
plt.grid()
plt.show()    


#complexity of model decreases as we increase reg. strength lambda
plt.figure(figsize=(8,8))
plt.semilogx(lambda_interval, coefficient_norm,'k')
plt.ylabel('L2 Norm')
plt.xlabel('Regularization strength, $\log_{10}(\lambda)$')
plt.title('Parameter vector L2 norm')
plt.grid()
plt.show()    

'''
Note: 
As one can see, the minimum test error is 26.49% which is quite high.
This is due to the fact that we have used all the attributes and only the regularization term for controlling model complexity. 
This means our model will classify y correctly about 1/4 times. 

'''