import numpy as np

import sklearn as sk
from sklearn import datasets

from MySVM import MySVM
from my_cross_val import my_cross_val

def print_results(title: str, results: list) -> None:
    print(title)
    print('Error rate per fold:',results)
    print('Mean error rate: {:.4f}'.format(results.mean()))
    print('Stdev error rate: {:.4f}\n'.format(results.std()))

## Boston Housing dataset ######### 
X, t = datasets.load_boston(return_X_y=True)

X = np.concatenate((X,np.ones((len(X),1))),axis=1)

t_75 = np.percentile(t, 75)

# for boston75
r = np.zeros(t.shape)
r[t>=t_75]=1
r[t<t_75]=-1

N = X.shape[0] # number of samples in X
d = X.shape[1] # number of features in X

### lambda = 0 ###
lambda_val = 0
model = MySVM(d, lambda_val=lambda_val, max_iter=2000, eta=1e-4)
results = my_cross_val(model,X,r,k=5)
print_results('My SVM: lambda={}'.format(lambda_val), results)
ftable = open("Table.txt", "a")
ftable.write('My SVM: lambda= ' + str(lambda_val) + "\t" + str(results) + "\t" + str(results.mean()) + "\t"  + str(results.std())+ "\n")
ftable.close()


#### lambda = 0.1 ###
lambda_val = 0.1
model = MySVM(d, lambda_val=lambda_val, max_iter=2000, eta=1e-5)
results = my_cross_val(model,X,r,k=5)
print_results('My SVM: lambda={}'.format(lambda_val), results)
# Write the data on a table text file
ftable = open("Table.txt", "a")
ftable.write('My SVM: lambda= ' + str(lambda_val) + "\t" + str(results) + "\t" + str(results.mean()) + "\t"  + str(results.std())+ "\n")
ftable.close()

#### lambda = 1000 ###
lambda_val = 1000
model = MySVM(d, lambda_val=lambda_val, max_iter=2000, eta=1e-7)
results = my_cross_val(model,X,r,k=5)
print_results('My SVM: lambda={}'.format(lambda_val), results)
ftable = open("Table.txt", "a")
ftable.write('My SVM: lambda= ' + str(lambda_val) + "\t" + str(results) + "\t" + str(results.mean()) + "\t"  + str(results.std())+ "\n")
ftable.close()