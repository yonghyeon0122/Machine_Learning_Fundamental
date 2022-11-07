# -*- coding: utf-8 -*-
"""
CSCI 5521 HW2
Yong Hyeon Yi
"""

# from my_cross_val import *
# from my_kmeans import *
import numpy as np   

"""=================================================================
multiGaussClassify
    k: the number of class
    d: the dimensionility of features
    diag: if the covariance is a diagonal matrix, diag = True 
    
fit(X, r) - Get estimates
    X: a feature matrix
    r: a class label 
    
predict(X) - Compare the discriminant function to predict the class
    X: a feature matrix
    
labelConfig(r)
    Initializing ri value, which depends on the classes
    If belongs to the calss, ri = 1. If not, ri = 0

priorEstimate(label_vector)
meanEstimate(X, label_vector)
covEstimate(X, mean_est, label_vector)
    Functions for calculating estimates to fit the data
    
discriminant(x, prior_est, mean_est, cov_est, class_index)
    returns a discriminant for each set of feature 'x' and class 'class_index'
    
===================================================================="""
class MultiGaussClassify:
    def __init__(self, k, d, diag):
        self.k = k
        self.d = d
        self.diag = diag
        self.label_vector = 0
        self.prior_est = 0
        self.mean_est = 0
        self.cov_est = 0
          
    def labelConfig(self, r):
        label_vector = np.zeros((self.k, len(r)))
        
        for index, label in enumerate(r):
            if(label == 0):
                label_vector[0][index] = 1
            elif(label == 1):
                label_vector[1][index] = 1 
                
        return label_vector
        
    def priorEstimate(self, label_vector):
                      
        prior_est = np.mean(label_vector, axis=1)                   
        
        return prior_est
    
    def meanEstimate(self, X, label_vector):
        
        normalizor = np.sum(label_vector, axis =1)
        mean_est = np.zeros((self.k, self.d))      
        
        for class_index, means in enumerate(mean_est):
            for feature_index, x in enumerate(X):
                mean_est[class_index] += label_vector[class_index][feature_index] * x 
            
            mean_est[class_index] = mean_est[class_index] / normalizor[class_index]
          
        
        return mean_est, normalizor
    
    def covEstimate(self, X, mean_est, label_vector):
        
        normalizor = np.sum(label_vector, axis =1)
        cov_est = np.zeros((self.k, self.d, self.d))
        
        for class_index, cov in enumerate(cov_est):
            for feature_index, x in enumerate(X):
                cov_est[class_index] += label_vector[class_index][feature_index] * np.matmul((x - mean_est[class_index]).reshape(-1,1), (x - mean_est[class_index]).reshape(1,-1)) 
                                                                                                          
            cov_est[class_index] = cov_est[class_index] / normalizor[class_index]
                    
        return cov_est
    
    
    def discriminant(self, x, prior_est, mean_est, cov_est, class_index):
        
        g = -1/2 * np.log(np.linalg.det(cov_est[class_index])) - 1/2 * self.matrixMult((x-mean_est[class_index]).reshape(1,-1),
                                                                                  np.linalg.inv(cov_est[class_index]),
                                                                                  (x-mean_est[class_index]).reshape(-1,1)) + np.log(prior_est[class_index])                                                                         
        
        
        return g
    
    def matrixMult(self, A, B, C):
        temp = np.matmul(A, B)
        
        return np.matmul(temp, C)
    
    def fit(self, X, r):
        
        self.label_vector = self.labelConfig(r)        
        self.prior_est = self.priorEstimate(self.label_vector)
        self.mean_est, self.normalizor = self.meanEstimate(X, self.label_vector)
        self.cov_est = self.covEstimate(X, self.mean_est, self.label_vector)          
        if(self.diag == True):
            for class_index in range(self.k):
                self.cov_est[class_index] = np.diag(np.diag(self.cov_est[class_index]))
            
        
    def predict(self, X):
        predict = np.zeros(len(X))
        g_list = np.zeros((self.k, (len(X))))
        discriminant_list = np.zeros((self.k, (len(X))))
        
        for class_index in range(self.k):
            for feature_index, x in enumerate(X):
                g = self.discriminant(x, self.prior_est, self.mean_est, self.cov_est, class_index)
                discriminant = np.exp(g)
                g_list[class_index][feature_index] = g
                discriminant_list[class_index][feature_index] = discriminant
            
        predict = np.argmax(g_list, axis=0)               
        
        return predict
                
    