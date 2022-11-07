import random
import numpy as np

from sklearn.datasets import load_digits
from my_cross_val import my_cross_val

from MyBagging import MyBagging
from MyRandomForest import MyRandomForest
from MyAdaboost import MyAdaboost

# for plot
import matplotlib.pyplot as plt
plt.switch_backend('agg')

def print_results(title: str, results: list) -> None:
    print(title)
    print('Error rate per fold:', results)
    print('Mean error rate: {:.4f}'.format(results.mean()))
    print('Stdev error rate: {:.4f}\n'.format(results.std()))

random.seed(2021)

X,t = load_digits(return_X_y=True)

class_0 = 3
class_1 = 8

X = X[(t==class_0) | (t==class_1)]
r = t[(t==class_0) | (t==class_1)]
r[r==class_0]=-1
r[r==class_1]=1

fig,ax = plt.subplots(4,2,figsize=(14, 14),sharey=True) # to plot images figsize[width, height]

N = X.shape[0] # number of samples in X
d = X.shape[1] # number of features in X

num_classifiers = 100  
num_folds = 5

#### Bagging

# max_depth = 1
my_bagging = MyBagging(num_classifiers,1)
results = my_cross_val(my_bagging,X,r,k=num_folds)
print_results('MyBagging with max_depth = 1', results[:,-1])

ax[0,0].plot(np.arange(num_classifiers),results.mean(axis=0))
ax[0,0].set_title('MyBagging with max_depth = 1')
ax[0,0].set_ylabel('Test Error')

# max_depth = None
my_bagging = MyBagging(num_classifiers,None)
results = my_cross_val(my_bagging,X,r,k=num_folds)
print_results('MyBagging with max_depth = None', results[:,-1])

ax[0,1].plot(np.arange(num_classifiers),results.mean(axis=0))
ax[0,1].set_title('MyBagging with max_depth = None')

print('**************\n')

#### Random Forest

# max_depth = 1, m = 1
my_rf = MyRandomForest(1,num_classifiers,1)
results = my_cross_val(my_rf,X,r,k=num_folds)
print_results('MyRandomForest with max_depth = 1 and m = 1', results[:,-1])

ax[1,0].plot(np.arange(num_classifiers),results.mean(axis=0))
ax[1,0].set_title('MyRandomForest with max_depth = 1 and m = 1')
ax[1,0].set_ylabel('Test Error')

# max_depth = 1, m = sqrt(d)
my_rf = MyRandomForest(int(np.sqrt(d)),num_classifiers,1)
results = my_cross_val(my_rf,X,r,k=num_folds)
print_results('MyRandomForest with max_depth = 1 and m = sqrt(d)', results[:,-1])

ax[1,1].plot(np.arange(num_classifiers),results.mean(axis=0))
ax[1,1].set_title('MyRandomForest with max_depth = 1 and m = sqrt(d)')

# max_depth = None, d = 1
my_rf = MyRandomForest(1,num_classifiers,None)
results = my_cross_val(my_rf,X,r,k=num_folds)
print_results('MyRandomForest with max_depth = None and m = 1', results[:,-1])

ax[2,0].plot(np.arange(num_classifiers),results.mean(axis=0))
ax[2,0].set_title('MyRandomForest with max_depth = None and m = 1')
ax[2,0].set_ylabel('Test Error')

# max_depth = None, sqrt(d)
my_rf = MyRandomForest(int(np.sqrt(d)),num_classifiers,None)
results = my_cross_val(my_rf,X,r,k=num_folds)
print_results('MyRandomForest with max_depth = None and m = sqrt(d)', results[:,-1])

ax[2,1].plot(np.arange(num_classifiers),results.mean(axis=0))
ax[2,1].set_title('MyRandomForest with max_depth = None and m = sqrt(d)')

print('**************\n')

#####  Adaboost

# max_depth = 1
my_adaboost = MyAdaboost(num_classifiers,1)
results = my_cross_val(my_adaboost,X,r,k=num_folds)
print_results('MyAdaboost with max_depth = 1', results[:,-1])

ax[3,0].plot(np.arange(num_classifiers),results.mean(axis=0))
ax[3,0].set_title('MyAdaboost with max_depth = 1')
ax[3,0].set_ylabel('Test Error')

# max_depth = None
my_adaboost = MyAdaboost(num_classifiers,None)
results = my_cross_val(my_adaboost,X,r,k=num_folds)
print_results('MyAdaboost with max_depth = None', results[:,-1])

ax[3,1].plot(np.arange(num_classifiers),results.mean(axis=0))
ax[3,1].set_title('MyAdaboost with max_depth = None')

ax[3,1].set_xlabel('No. of Ensemble Trees')

fig.subplots_adjust(left=0.05, right=0.95, top=0.95, bottom=0.05,wspace=0.3,hspace=0.3)
fig.savefig('hw3q2.png')
