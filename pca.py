import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import svd
from continuous_data import *


Y = (X_cont - np.ones((N_cont, 1)) * X_cont.mean(axis=0)) / X_cont.std(axis=0)

# PCA by computing SVD of Y
U, S, Vh = svd(Y, full_matrices=False)
# scipy.linalg.svd returns "Vh", which is the Hermitian (transposed)

# Compute variance explained by principal components
rho = (S*S) / (S*S).sum() 

threshold = 0.9

# Plot variance explained
plt.figure()
plt.plot(range(1, len(rho)+1), rho, 'x-')
plt.plot(range(1, len(rho)+1), np.cumsum(rho), 'o-')
plt.plot([1, len(rho)], [threshold, threshold], 'k--')
plt.title('Variance explained by principal components')
plt.xticks(range(1, len(rho)+1))
plt.xlabel('Principal component')
plt.ylabel('Variance explained')
plt.legend(['Individual', 'Cumulative', 'Threshold'])
plt.grid()
plt.show()

'''
First 2 PCAs describe roughly 90% of variance
'''

V = Vh.T

# Project the centered data onto principal component space
Z = Y @ V  # @ in python: projects vector

# Indices of the principal components to be plotted: v1 and v2
i = 0
j = 1

# Plot PCA of the data
f = plt.figure()
plt.title('Heart Attack Possibility data: PCA')
# Z = array(Z)
for c in range(C):
    # select indices belonging to class c:
    class_mask = y == c
    plt.plot(Z[class_mask, i], Z[class_mask, j], 'o', alpha=.5)
plt.legend(classNames)
plt.xlabel('PC{0}'.format(i+1))
plt.ylabel('PC{0}'.format(j+1))

# Output result to screen
plt.show()
'''
Projection of attributes on PC1
The first 2 components explained 90% of variance.
So we will select v1,v2 and we'll look at their coefficients:
    
'''
pcs = [0, 1, 2, 3]
legendStrs = ['PC'+str(e+1) for e in pcs]
c = ['r', 'g']
bw = .2
r = np.arange(1, M_cont+1)
for i in pcs:    
    plt.bar(r+i*bw, V[:, i], width=bw)
plt.xticks(r+bw, cont_attr)
plt.xlabel('Attributes')
plt.ylabel('Component coefficients')
plt.legend(legendStrs)
plt.grid()
plt.title('Heart Attack Possibility: PCA Component Coefficients')
plt.show()

'''
Inspecting the plot, we see that the first PC has a large negative magnitude 
for chol, whereas the 2nd PC has a large negative magnitude for thalach.  
'''
print('\n PC1:')
print(V[:, 0].T)
print('\n PC2:')
print(V[:, 1].T)
print('\n PC3:')
print(V[:, 2].T)
print('\n PC4:')
print(V[:, 3].T)

# Looking at the data for target class 0:
less_chance_data = Y[y == 0, :]

print('\n First observation for a class with less chance of heart attack:')
print(less_chance_data[0, :])  # selects first observation

# Based on the coefficients and the attribute values for the observation
# displayed, one would expect the projection onto PC1 and PC2 to be positive:

# You can determine the projection by (remove comments):
print('\n - its projection onto PC1 ')
print(less_chance_data[0, :]@V[:, 0])

print('\n - its projection onto PC2')
print(less_chance_data[0, :]@V[:, 1])
print()

'''
This makes sense. 
One would expect the printed observation to have a negative 
projection onto PC1 since it has positive age, positive trestbps, positive chol,
negative thalach and positive oldpeak.

One would also expect the printed observation to hace a positive projection
onto PC2 based on the attributes mentioned above (postivie age, trestbps etc.)


'''
