# -*- coding: utf-8 -*-
"""
Created on Sat Nov 14 19:41:58 2020

@author: bejen
"""


from sklearn import model_selection

cvf = 10
K=10
# splitting is done using stratified fold so classes are equally balanced
# inner cross-validation:
CV1 = model_selection.StratifiedKFold(cvf, shuffle=True, random_state = 0) 

#outer cross-validation
CV2 = model_selection.StratifiedKFold(K, shuffle=True, random_state = 0) 