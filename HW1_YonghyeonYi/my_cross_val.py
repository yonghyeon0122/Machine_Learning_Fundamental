# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 22:25:16 2021

@author: yi000055
"""

import numpy as np
#import matplotlib.pyplot as plt
from sklearn.datasets import load_boston
from sklearn.datasets import load_digits
from sklearn.svm import LinearSVC
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

from sklearn.model_selection import KFold

"""=============================================
boston_dataset()
    X: features per sample
    t: price information per sample
    percentile: decides class r

tow: percentile

return
    r: class (depends on the percentile)
============================================="""
def boston_dataset(X, t, percentile):
    tow = np.percentile(t, percentile)
    r = np.array([], dtype= int )
    count=0
    
    for price in t:
        if price >= tow:
            r = np.append(r, [1])
            count += 1
        else:
            r = np.append(r, [0])
        
    prior_1 = count/len(t)
    prior_0 = 1-count/len(t)

    print("=====Boston" , str(percentile) + "======")        
    print("the " , str(percentile) , "th percentile is " , str(tow))
    print("p(r=1) is " , str(prior_1))
    print("p(r=0) is " , str(prior_0))
    
    f = open("Result.txt", "a")
    f.write("=====Boston" + str(percentile) + "======\n")
    f.write("the " + str(percentile) + "th percentile is " + str(tow) + "\n")
    f.write("p(r=1) is " + str(prior_1) + "\n")
    f.write("p(r=0) is " + str(prior_0) + "\n")
    f.close()
    
    return r

"""=============================================
q4()
    dataset preparing function
return
    Xboston
    r_boston50
    r_boston75
    X_digits
    r_digits        
============================================="""
def q4():
    X_boston, t_boston = load_boston(return_X_y=True)
    X_digits, r_digits = load_digits(return_X_y=True)  

    r_boston50 = boston_dataset(X_boston, t_boston, 50)
    r_boston75 = boston_dataset(X_boston, t_boston, 75) 

    print("=======The number of samples========")
    print("Boston: " + str(len(r_boston50)))
    print("Digits: " + str(len(r_digits)))
    
    return X_boston, r_boston50, r_boston75, X_digits, r_digits

"""=============================================
my_cross_val()
    method: one of LinearSVC, SVC, and LogisticRegression
    X: feature vector
    r: class vector
    k: the number of fold
    
return
    error_rate_all, error_mean, error _std
============================================="""
def my_cross_val(method, X, r, k):
        
    if (method == "LinearSVC"):
        clf = LinearSVC(max_iter=2000)
    elif (method == "SVC"):
        clf = SVC(gamma='scale', C=10)
    else:
        clf = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000)
        
    f = open("Result.txt", "a")
    
    #KFold() is used to split the dataset into k pieces
    kf = KFold(n_splits=k)
    k=0
    error_rate_all = np.array([])
            
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        r_train, r_test = r[train_index], r[test_index]
        
        clf.fit(X_train, r_train) # fit the training set
        predict = clf.predict(X_test) # predict the test set after fitting
        score = clf.score(X_test, r_test) # ratios of the correct prediction
        error_rate = 1 - score  # ratios of the incorrect prediction
        error_rate_all = np.append(error_rate_all, [error_rate])
        k += 1  
        print("Fold ", k, ": ", error_rate )
        f.write("Fold " + str(k) + ": " + str(error_rate) + "\n" )
    error_mean = np.mean(error_rate_all) # mean of the error rate
    error_std = np.std(error_rate_all) # standard deviation of the error rate
    
    print("Mean: ", error_mean)
    f.write("Mean: " + str(error_mean) + "\n")
    print("Standard Deviation: ", error_std)
    f.write("Standard Deviation: " + str(error_std) + "\n")
    
    f.close()
    
    return error_rate_all, error_mean, error_std
    