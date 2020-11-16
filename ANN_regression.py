import numpy as np
import torch
from scipy.stats import stats
from sklearn.preprocessing import OneHotEncoder
from toolbox_02450 import train_neural_net, draw_neural_net

from auxiliary_functions import rlr_validate, network_validate_regression, CV4
import matplotlib.pyplot as plt

from data_aquisition import *

enc = OneHotEncoder()
# we get our discrete variables: sex, cp, slope and target
X_discrete = np.stack((X_sel[:, 1], X_sel[:, 2], X_sel[:, 7], y), axis=-1)
enc.fit(X_discrete)
X_enc = enc.transform(X_discrete).toarray()
# stack on top of the continuous ones
y = X_cont[:,3].T
y_label = "thalach"
X_cont = np.delete(X_cont, 3, axis=1)
X = np.column_stack((X_cont, X_enc))
N, M = X.shape
attributeNames = np.array(['age', 'trestbps', 'chol', 'oldpeak',
                           'female', 'male', 'typical_cp', 'atypical_cp',
                           'no_cp', 'asymptomatic_cp', 'slope_up', 'slope_flat', 'slope_down', 'target0', 'target1'],
                          dtype='<U8')
y = np.asmatrix(y).reshape(N, 1)
X = stats.zscore(X)

do_pca_preprocessing = False
if do_pca_preprocessing:
    Y = stats.zscore(X,0)
    U,S,V = np.linalg.svd(Y,full_matrices=False)
    V = V.T
    #Components to be included as features
    k_pca = 3
    X = X @ V[:,:k_pca]
    N, M = X.shape

K = 10
h_interval = np.array([1, 2, 3, 4, 5])
max_iter = 50000 #this has to be as high as possible to make a good prediction, however it takes longer computation time
n_replicates = 1 #due to computation time

errors = []   # a list for storing generalizaition error after each outer cv-fold
h_optimal_list = []     # a list for storing optimal hidden units no after each outer cv-fold
ANN_best_models = []    # a list for models for storing models after each outer cv-fold

# Make figure for holding summaries (errors and learning curves)
summaries, summaries_axes = plt.subplots(1,2, figsize=(10,5))
# Make a list for storing assigned color of learning curve for up to K=10
color_list = ['tab:orange', 'tab:green', 'tab:purple', 'tab:brown', 'tab:pink',
              'tab:gray', 'tab:olive', 'tab:cyan', 'tab:red', 'tab:blue']

opt_val_err, opt_n_h_units = network_validate_regression(X, y, h_interval)


for k, (train_index, test_index) in enumerate(CV4.split(X, y)):
    print('\nCROSSVALIDATION OUTER FOLD: {0}/{1}'.format(k + 1, K))

    # network_validate_classification does the inner cross-validation with cvf=10
    opt_val_err, opt_n_h_units = network_validate_regression(X, y, h_interval)

    model = lambda: torch.nn.Sequential(
        torch.nn.Linear(M, opt_n_h_units),  # M features to H hiden units
        torch.nn.Tanh(),  # 1st transfer function,
        torch.nn.Linear(opt_n_h_units, 1),  # H hidden units to 1 output neuron
          # no final tranfer function
    )
    loss_fn = torch.nn.MSELoss()

    X_train = torch.Tensor(X[train_index, :])
    y_train = torch.Tensor(y[train_index])
    X_test = torch.Tensor(X[test_index, :])
    y_test = torch.Tensor(y[test_index])

    # Train the net on training data
    net, final_loss, learning_curve = train_neural_net(model,
                                                       loss_fn,
                                                       X=X_train,
                                                       y=y_train,
                                                       n_replicates=n_replicates,
                                                       max_iter=max_iter)

    print('\n\tBest loss: {}\n'.format(final_loss))
    
    # Determine estimated class labels for test set
    # y predicted
    y_test_est = net(X_test)

    # Determine errors and errors
    se = (y_test_est.float()-y_test.float())**2 # squared error
    mse = (sum(se).type(torch.float)/len(y_test)).data.numpy() #mean
    errors.append(mse) # store error rate for current CV fold  # store error rate for current CV fold

    h_optimal_list.append(opt_n_h_units)
    ANN_best_models.append(net)

    # Display the learning curve for the best net in the current fold
    h, = summaries_axes[0].plot(learning_curve, color=color_list[k])
    h.set_label('CV fold {0}'.format(k+1))
    summaries_axes[0].set_xlabel('Iterations')
    summaries_axes[0].set_xlim((0, max_iter))
    summaries_axes[0].set_ylabel('Loss')
    summaries_axes[0].set_title('Learning curves')

errors = np.asarray(errors)
errors_sqrt =np.round(np.sqrt((np.asarray(errors))),4)
opt_index_2 = np.argmin(errors_sqrt)
opt_val_err = np.min(errors_sqrt)
opt_n_h_units = h_optimal_list[opt_index_2]
best_ANN_model = ANN_best_models[opt_index_2]

# Display the MSE across folds
summaries_axes[1].bar(np.arange(1, K+1), np.squeeze(np.asarray(errors_sqrt)), color=color_list)
summaries_axes[1].set_xlabel('Fold')
summaries_axes[1].set_xticks(np.arange(1, K+1))
summaries_axes[1].set_ylabel('MSE')
summaries_axes[1].set_title('Test mean-squared-error')

print('Diagram of best neural net:')
weights = [best_ANN_model[i].weight.data.numpy().T for i in [0, 2]]
biases = [best_ANN_model[i].bias.data.numpy() for i in [0, 2]]
tf = [str(best_ANN_model[i]) for i in [1, 2]]
draw_neural_net(weights, biases, tf, attribute_names=attributeNames.tolist())

# When dealing with regression outputs, a simple way of looking at the quality
# of predictions visually is by plotting the estimated value as a function of 
# the true/known value - these values should all be along a straight line "y=x", 
# and if the points are above the line, the model overestimates, whereas if the
# points are below the y=x line, then the model underestimates the value
plt.figure(figsize=(10,10))
y_est = y_test_est.data.numpy(); y_true = y_test.data.numpy()
axis_range = [np.min([y_est, y_true])-1,np.max([y_est, y_true])+1]
plt.plot(axis_range,axis_range,'k--')
plt.plot(y_true, y_est,'ob',alpha=.25)
plt.legend(['Perfect estimation','Model estimations'])
plt.title('Predicted thalach: estimated versus true value (for last CV-fold)')
plt.ylim(axis_range); plt.xlim(axis_range)
plt.xlabel('True value')
plt.ylabel('Estimated value')
plt.grid()

plt.show()

print("Test errors for ANN per k fold are: ", np.round((np.asarray(errors_sqrt)),4))
print("With ptimal hidden units for each fold: ", h_optimal_list)
# Print the minimum error rate for the optimal fold:

#print('\nEstimated generalization error, RMSE: {0}'.format(round(np.sqrt(np.mean(errors)), 4)))
print("Optimal test error is {}% with no of hidden units {}".format(np.round(opt_val_err * 100, 2), opt_n_h_units))


