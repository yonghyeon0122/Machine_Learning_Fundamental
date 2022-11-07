# -*- coding: utf-8 -*-
"""
CSCI 5521 HW2
Yong Hyeon Yi
"""

import numpy as np

def meanProjAndS(X):
    
    d = len(X[0])
    S = np.zeros((d, d))
    normalizor = len(X)
    mean_prj = np.mean(X, axis=0)
    
    for sample_index, x in enumerate(X):
        S += np.matmul((x - mean_prj).reshape(-1,1), (x - mean_prj).reshape(1,-1))
    S = S / normalizor
    
    
    
    
    return mean_prj, S
        


"""=================================================================
Principal Component Analysis

myPCA(X, k)
    X: the original data (N by d) from the Digits dataset
    k: how many principle components used for projection (k=2 in this case)

    return
    W: the projection matrix (d by 2)
    mu_est: the etimated mean (d by 1)
 
ProjectDatapoints(X, W, mu_est)
    X: the original data (N by d)
    W: from myPCA (d by 2)
    
    return
    X_prj: the projected data    
===================================================================="""
def myPCA(X, k):
    
    # projected mean and S value
    W = np.zeros((len(X[0]), k)) # the list of eigen values
    mean_prj, S = meanProjAndS(X)
    
    # the first two largest eigenvectors with the largest eigenvalue
    # the two eigenvectors are used as a projection matrix
    eigenW, eigenV = np.linalg.eig(S)
    sortedIndex = np.argsort(eigenW)
    for index in range(k):
        W[:, index] = eigenV[:,sortedIndex[-(index+1)]] 
    W = np.transpose(W)
    
    
    # estimated mean
    mu_est = np.mean(X, axis=0)
    
    return W, mu_est

def ProjectDatapoints(X, W, mu_est):
    
    # projected feature vector, X_prj
    # each samples are multiplied by W to generate X_prj
    # features are subtracted by mu_est before projection
    X_prj = np.zeros((len(W), len(X)))
    
    for sample_index, x in enumerate(X):
          x_prj = np.matmul(W, (x - mu_est).reshape(-1, 1))
          X_prj[:, sample_index] = x_prj.reshape(-1)
  
     
    X_prj = np.transpose(X_prj)
    
    return X_prj
