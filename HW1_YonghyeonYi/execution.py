# -*- coding: utf-8 -*-
"""
CSCI 5521 HW1
Yong Hyeon Yi
"""

from my_cross_val import *
#import matplotlib.pyplot as plt

"""=============================================
execution()
    method: LinearSVC / SVC / LogisticRegression
    X: feature vector
    r: Class
    k: the number of fold
    dataset: the name of the dataset
============================================="""
def execution(method, X, r, k, dataset):
    f = open("Result.txt", "a")
    print("==== Error rates for ", method, " with ", dataset, "====")
    f.write("==== Error rates for " + method  + " with " + dataset + "====" + "\n")
    f.close()
    error_rate_all, error_mean, error_std = my_cross_val(method, X, r, k)
    table_generation(method, X, r, k, dataset, error_rate_all, error_mean, error_std)
    
    return 0

def table_generation(method, X, r, k, dataset, error_rate_all, error_mean, error_std):
    ftable = open("Table_data.txt", "a")
    ftable.write("==== Error rates for " + method  + " with " + dataset + "====" + "\n")
    
    for n in range(1, k+1):
        ftable.write("F" + str(n) + "\t")        
    ftable.write("Mean\t")
    ftable.write("SD\n")
    
    for error_rate in error_rate_all:
        ftable.write("%0.4f\t" % error_rate)
    ftable.write("%0.4f\t" % error_mean)
    ftable.write("%0.4f\n\n" % error_std)
        
    ftable.close()
    
    return 0
"""=============================================
Boston50
    Feature vector: X_boston
    Class vector: r_boston50
Boston70
    Feature vector: X_boston
    Class vector : r_boston75
Digits
    Feature vector: X_digits
    Class vector: r_digits
============================================="""
f = open("Result.txt", "w")
f.close()
ftable = open("Table_data.txt", "w")
ftable.close()

X_boston, r_boston50, r_boston75, X_digits, r_digits = q4()

"""=============================================
k fold cross validation using
    method_1: LinearSVC
    method_2: SVC
    method_3: LogisticRegression
============================================="""
method_1 = "LinearSVC"
method_2 = "SVC"
method_3 = "LogisticRegression"

dataset_1 = "Boston50"
dataset_2 = "Boston75"
dataset_3 = "Digits"

execution(method_1, X_boston, r_boston50, 10, dataset_1)
execution(method_1, X_boston, r_boston75, 10, dataset_2)
execution(method_1, X_digits, r_digits, 10, dataset_3)

execution(method_2, X_boston, r_boston50, 10, dataset_1)
execution(method_2, X_boston, r_boston75, 10, dataset_2)
execution(method_2, X_digits, r_digits, 10, dataset_3)

execution(method_3, X_boston, r_boston50, 10, dataset_1)
execution(method_3, X_boston, r_boston75, 10, dataset_2)
execution(method_3, X_digits, r_digits, 10, dataset_3)

