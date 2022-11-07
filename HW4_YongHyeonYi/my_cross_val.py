import numpy as np

import sklearn as sk
from sklearn.model_selection import KFold



def my_cross_val(method,X,r,k=5):

    kf = KFold(n_splits=k, random_state=None, shuffle=False)

    results = []
    
    for train_index, test_index in kf.split(X):

        X_train, X_test = X[train_index], X[test_index]
        r_train, r_test = r[train_index], r[test_index]

        method.fit(X_train,r_train)
        
        r_pred = method.predict(X_test)

        test_err = (r_pred!=r_test).sum()/len(r_pred)

        results.append(test_err)

    results = np.asarray(results)

    return results






















    

    


    








#X,r = datasets.load_digits(return_X_y=True)


# Prefer dual=False when n_samples > n_features.
#clf = LinearSVC(max_iter=2000)#,dual=False,) 
#clf = LinearSVC(max_iter=2000,random_state=0) 
#clf = SVC(gamma='scale', C=10) 
#clf = LogisticRegression(penalty='l2', solver='lbfgs', multi_class='multinomial',max_iter=5000)


##
##n_samples = len(r)
##k=10
##test_size = int(np.ceil(n_samples/k))
##
##indices = np.arange(n_samples) # create the index
##
##results = []
##
##for i in range(0,n_samples,test_size):
##    train_mask = np.ones(n_samples, dtype=bool)
##
##    test_index = indices[i:i+test_size]
##    train_mask[test_index]=0
##
##    train_index = indices[train_mask]
##
##    X_train = X[train_index]
##    y_train = r[train_index]
##
##    X_test = X[test_index]
##    y_test = r[test_index]
##
##
##    clf.fit(X_train,y_train)
##
##    y_pred = clf.predict(X_test)
##
##    test_err = (y_pred!=y_test).sum()/len(y_pred)
##
##    results.append(test_err)
##
##results = np.asarray(results)
##
##for r in results:
##    print('{:.4f}&'.format(r))
##print('{:.4f}&{:.4f}'.format(results.mean(),results.std()))
##
##
