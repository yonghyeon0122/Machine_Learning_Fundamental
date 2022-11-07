import numpy as np

import sklearn as sk
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

from MultiGaussClassify import MultiGaussClassify

from my_cross_val import my_cross_val
from TableGeneration import *




X, t = datasets.load_boston(return_X_y=True)

t_50 = np.percentile(t, 50)

# for boston50
r = np.zeros(t.shape)
r[t>=t_50]=1


N = X.shape[0] # number of samples in X
d = X.shape[1] # number of features in X


### YOUR CODE STARTS HERE ###

"""================================================
The mean vector and the covariance matrix 
    Xt: transposed dataset
    mean: average of each feature
    cov: covariance of each feature
==================================================="""
MultiGaussClassify_fullCov = MultiGaussClassify(2, d, False)
MultiGaussClassify_diagCov = MultiGaussClassify(2, d, True)
LogisticRegression = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial', max_iter=5000)

# label_vector = MultiGaussClassify_fullCov.labelConfig(r)
# prior_est = MultiGaussClassify_fullCov.priorEstimate(label_vector)
# mean_est, normalizor =  MultiGaussClassify_fullCov.meanEstimate(X, label_vector)
# cov_est =  MultiGaussClassify_fullCov.covEstimate(X, mean_est, label_vector)

# MultiGaussClassify_fullCov.fit(X, r)
# predict, g_list, discriminant_list = MultiGaussClassify_fullCov.predict(X)

result1 = my_cross_val(MultiGaussClassify_fullCov, X, r)
mean1, std1 = meanAndStd(result1)
result2 = my_cross_val(MultiGaussClassify_diagCov, X, r)
mean2, std2 = meanAndStd(result2)
result3 = my_cross_val(LogisticRegression, X, r)
mean3, std3 = meanAndStd(result3)

## Write the data into a Table_data.txt file
table_name = "Table_hw2q2.txt"
ftable = open(table_name, "w")
result_name = "Result_hw2q2.txt"
fresult = open(result_name, "w")
ftable.close()
fresult.close()
k=5

resultPrint("MultiGaussClassify Full Cov", k, "Boston50", result1, mean1, std1, result_name)
resultPrint("MultiGaussClassify Diag Cov", k, "Boston50", result2, mean2, std2, result_name)
resultPrint("LogisticRegression", k, "Boston50", result3, mean3, std3, result_name)
table_generation("MultiGaussClassify Full Cov", k, "Boston50", result1, mean1, std1, table_name)
table_generation("MultiGaussClassify Diag Cov", k, "Boston50", result2, mean2, std2, table_name)
table_generation("LogisticRegression", k, "Boston50", result3, mean3, std3, table_name)




