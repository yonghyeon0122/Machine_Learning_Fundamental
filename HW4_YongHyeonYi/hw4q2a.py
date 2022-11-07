import numpy as np

import sklearn as sk
from sklearn import datasets

from MyLogisticRegression import MyLogisticRegression
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

N = X.shape[0] # number of samples in X
d = X.shape[1] # number of features in X

model = MyLogisticRegression(d,max_iter=5000,eta=1e-3) # max_iter and eta has been changed for better convergence
results = my_cross_val(model,X,r,k=5)
print_results('My Logistic Regression', results)
# Write the result in the table text file
ftable = open("Table.txt", "w")
ftable.write("My Logistic Regression " + "\t" + str(results) + "\t" + str(results.mean()) + "\t"  + str(results.std())+ "\n")
ftable.close()