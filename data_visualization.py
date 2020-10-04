# -*- coding: utf-8 -*-

from main import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import (yticks, plot, boxplot, xticks, ylabel, title, imshow, 
                               figure, subplot, hist, xlabel, ylim, show, legend, cm, colorbar)
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import zscore

# DATA VIZUALIZATION ###############################################

#Histograms  
figure(figsize=(8,7))
u = np.floor(np.sqrt(M)); v = np.ceil(float(M)/u)
for i in range(M):
    subplot(u,v,i+1)
    hist(X[:,i], color=(0.1, 0.8-i*0.1, 0.4))
    xlabel(attributeNames[i])
    ylim(0,N/2)
    
show()
'''
Seems like age, chol and thalach are normally distributed
'''

# 

# BOXPLOT: for all attributes ################################################

boxplot(X)
xticks(range(1,9),attributeNames)
title('Boxplot for all attributes')
show()

'''
It can clearly be seen that tje continuous variables are at different scales
(e.g.: cholesterol is between 100-400 whereas olpeak is from 0 to 6)
This will be a problem for applying ML algorithms, 
especially for Classification problems where Euclidian distance is used


We will substract the mean and divide by st. dev for the continous variables:
chol, thalach, oldpeak

Note: Age can be left out of this since we might transform it into a discrete variable
later - for making a better model - for example we could use bining but this is beyond the scope
of this project.

'''
#create matrix of continuous variables: chol, thalach, oldpeak
X_cont = np.column_stack((X[:,3],X[:,4],X[:,5]))

attributeNames_cont = np.array([attributeNames[3],attributeNames[4],attributeNames[5]])

# standardize X_cont (relative to mean and standard deviation)
Y2 = zscore(X_cont, ddof=1)
'''
another way:
Y2 = X_cont - np.ones((N, 1))*X_cont.mean(0)
Y2 = Y2*(1/np.std(Y2,0))
'''

# Replace standardized columns into X
X_stand = X
X_stand[:,3] = Y2[:,0]
X_stand[:,4] = Y2[:,1]
X_stand[:,5] = Y2[:,2]
X_stand = np.delete(X_stand,0,1)

attributeNames_ageexcl = np.delete(attributeNames,0) 

# BOXPLOT: for all attributes EXCEPT AGE: standardized to check for outliers:
boxplot(X_stand)
xticks(range(1,9),attributeNames_ageexcl)
title('Boxplot to check for outliers (with standardized cont. values)')
show()

# BOXPLOT FOR STANARDIZED CONTINOUS VALUES ################################# 
boxplot(Y2)
xticks(range(1,4),attributeNames_cont)
title('Boxplot for Standardized Continuous Values')
show()

'''
Looks better now. WE SHOULD DISCUSS WHICH OUTLIERS TO REMOVE
'''

# BOXPLOT based on each class #################################################
# more chance/ less chance of heart attack
# Also aligned on the same axis for better comparison

figure(figsize=(14,7))
for c in range(C):
    subplot(1,C,c+1)
    class_mask = (y==c)     
    boxplot(Y2[class_mask,:])
    title('Boxplot for Class: '+classNames[c])
    xticks(range(1,len(attributeNames_cont)+1), [a[:7] for a in attributeNames_cont], rotation=45)
    y_up = Y2.max()+(Y2.max()-Y2.min())*0.1; y_down = Y2.min()-(Y2.max()-Y2.min())*0.1
    ylim(y_down, y_up)

show()



#SCATTERPLOT for continuous values ############################################

# X_cont2: has age, chol, thalach and oldpeak - not normalized
X_cont2 = np.column_stack((X[:,0],X_cont))
attributeNames_cont2 = np.array(['age','chol', 'thalach', 'oldpeak'])
N2, M2 = X_cont2.shape

# Scatterplots:
figure(figsize=(12,10))
for m1 in range(M2): #loops through attributes - x direction
    for m2 in range(M2):  #loops through attrbutes - y direction
        subplot(M2, M2, m1*M2 + m2 + 1)
        #loops through classes:
        for c in range(C):
            class_mask = (y==c)
            plot(np.array(X_cont2[class_mask,m2]), np.array(X_cont2[class_mask,m1]), '.')
      
            #LABELS:
            if m1==M2-1: #makes labels when 1st axis
                xlabel(attributeNames_cont2[m2])
            else:
                xticks([])
            if m2==0: #makes labels for y axis
                ylabel(attributeNames_cont2[m1])
            else:
                yticks([])

legend(classNames)

show()

# 3D PLOT ###################################################################
# Choice: age, chol and thalach

ind = [0, 1, 2]
colors = ['blue', 'green', 'red']

f = figure()
ax = f.add_subplot(111, projection='3d')
for c in range(C):
    class_mask = (y==c)
    s = ax.scatter(X_cont2[class_mask,ind[0]], X_cont2[class_mask,ind[1]], X_cont2[class_mask,ind[2]], c=colors[c])

ax.view_init(30, 220)
ax.set_xlabel(attributeNames_cont2[ind[0]])
ax.set_ylabel(attributeNames_cont2[ind[1]])
ax.set_zlabel(attributeNames_cont2[ind[2]])

show()

# MATRIX PLOT ###############################################################

# Uses standardized matrix of continous values: !Note: age is also standardized
Y2_2 = zscore(X_cont2, ddof=1)

# Replace standardized columns into X
X_stand2 = X
X_stand2[:,0] = Y2_2[:,0]
X_stand2[:,3] = Y2_2[:,1]
X_stand2[:,4] = Y2_2[:,2]
X_stand2[:,5] = Y2_2[:,3]

figure(figsize=(12,6))
imshow(X_stand2, interpolation='none', aspect=(8./N), cmap=cm.gray);
xticks(range(8), attributeNames)
xlabel('Attributes')
ylabel('Data objects')
title('Heart Attack Possibility: Data Matrix')
colorbar()

show()
