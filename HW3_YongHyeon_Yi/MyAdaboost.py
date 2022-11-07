import numpy as np
from scipy import stats
import random

from sklearn.tree import DecisionTreeClassifier


class MyAdaboost:
    def __init__(self, num_iters, max_depth=None):

        """
            num_iters: [int],
                Number of tree classifiers (i.e., number of iterations or rounds-the variable T from the lecture notes) in the ensemble.
                Must  be  an  integer >=1.

            max_depth: [int, default=None],
                The maximum depth of the trees. If None, then nodes are expanded until all leaves are pure.
        """
        ### Your code starts here ###
        self.num_iters = num_iters
        self.max_depth = max_depth
        self.clf_list = []
        self.clf_weight_list = []
        self.predict_tree_list = []


    def bootstrapDataset(self, X, r, weight_dist):
        # Make a bootstrap dataset
        # The random sampling is done based on the weight distribution
        # To avoid the 'probablilites do not sum to 1' error, the probabilites are normalized again
        X_bootstrap = np.zeros(np.shape(X))
        r_bootstrap = np.zeros(np.shape(r))
        weight_dist = weight_dist / np.sum(weight_dist)
        
        random_sample = np.random.choice(len(X), size = len(X), replace=True, p=weight_dist)
        
        for sample_index, random_sample_index in enumerate(random_sample):
            X_bootstrap[sample_index] = X[random_sample_index]
            r_bootstrap[sample_index] = r[random_sample_index]
                
        return X_bootstrap, r_bootstrap
    
    def fit(self, X, r):
        """
            Build a AdaBoost classifier from the training set (X, r)
            X: the feature matrix 
            r: class labels 
            
            1. Initialize the weight distribution as a uniform distribution, 1/N(sample number)
            2. Train a weak learner, the decision tree, using the weight distribution
            3. Get weak hypothesis Gt with error error_tree
            4. Choose alpha_t, which is a function of error_tree
            5. Update the weight for each sample basted on t the error and prediction
        """

        ### Your code starts here ###
        N = len(X) # the number of samples
        weight_dist = np.ones(N)/N # uniform distribution initially
        self.clf_list = []
        self.clf_weight_list = []
        
        
        for nth_iter in range(self.num_iters): 
            
            # Get a bootstrapped data
            X_bootstrap, r_bootstrap = self.bootstrapDataset(X, r, weight_dist)
            
            clf_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, max_depth = self.max_depth)
            clf_tree.fit(X_bootstrap, r_bootstrap)
            self.clf_list.append(clf_tree)
            
            # get the error rate
            # use the indicator function and the weight distribution
            # the indicator output is '1' if misclassified
            predict_tree = clf_tree.predict(X_bootstrap)
            indicator = np.multiply(predict_tree, r_bootstrap)
            indicator[indicator==1] = 0
            indicator[indicator==-1] = 1 
            
            error_tree = np.sum(np.multiply(weight_dist, indicator), axis=0) / np.sum(weight_dist, axis=0) + 1e-10
            clf_weight = 1/2 * np.log((1-error_tree) / error_tree)
            self.clf_weight_list.append(clf_weight)
            
            # update the next weight distribution
            weight_dist = np.multiply(weight_dist, np.exp(-1 * np.multiply(r_bootstrap, predict_tree))) / np.sum(weight_dist, axis=0)
              
    

    def predict(self, X):    

        """
            Predict class(es) for X as the number of tree classifiers in the ensemble grows

            X: the feature matrix

            Return:

            r_pred: [list], contains predictions as the number of tree classifiers in the ensemble grows
            The list should have the same size of num_trees.
            Each element in the list has the same dimension of X (number of data points in the test set),
            and the prediction is made based on the first k tree classifiers in the ensemble as k grows from 1 to num_trees.
            E.g., when k = 1, the Bagging classifier makes predictions only based on the first tree classifier you built from self.fit function;
            when k = num_tress, the Bagging classifier makes predictions based on all tree classifiers you built from self.fit function
            

        """
        ### Your code starts here ###
        # Comination of the classifiers makes the final classifier
        # Weighted sum of each iteration is the final output
        r_pred = []
        predict_final = np.zeros(len(X))        
                
        for k in range(self.num_iters):
            # make predictions based on the first k tree classifiers
            predict_final = np.zeros(len(X))
            for iter_index in range(k):
                predict_tree = self.clf_list[iter_index].predict(X)
                predict_final = np.add(np.multiply(self.clf_weight_list[k], predict_tree), predict_final)
            
            predict_final = np.sign(predict_final)
           
            r_pred.append(predict_final)
            
        return r_pred


