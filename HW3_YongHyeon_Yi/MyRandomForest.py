import numpy as np
from scipy import stats
import random


from sklearn.tree import DecisionTreeClassifier


class MyRandomForest:

    def __init__(self, num_features, num_trees, max_depth=None):

        """
            num_features: [int]
                Number of random features to choose at each split in the decision tree
                (the variable m from the lecture notes, and must be 1 <= num_features <= d),

            num_trees: [int],
                Number of tree classifiers in the ensemble (must be an integer >=1)


            max_depth: [int, default=None],
                The maximum depth of the trees. If None, then nodes are expanded until all leaves are pure.
        """
        ### Your code starts here ###
        self.num_features = num_features
        self.num_trees = num_trees
        self.max_depth = max_depth
        self.X_ensamble = 0
        self.r_ensamble = 0
        self.clf_ensamble = []
        self.feature_index_list = []
        
    def bootstrapDataset(self, X, r):
        # Make a bootstrap dataset
        
        X_bootstrap = np.zeros(np.shape(X))
        r_bootstrap = np.zeros(np.shape(r))
        random_sample = np.random.randint(len(X), size = len(X))
        
        for sample_index, random_sample_index in enumerate(random_sample):
            X_bootstrap[sample_index] = X[random_sample_index]
            r_bootstrap[sample_index] = r[random_sample_index]
                
        return X_bootstrap, r_bootstrap      

    def majorityVote(self, r_pred_ensamble):
        # select the list of the predictions for each sample
        # which is a column vector
        # for each column, get the majority vote result
        
        # First, add the elements in each column vector                
        r_pred_majorityVote = np.sum(r_pred_ensamble, axis = 0)
        # Then, get the sign of each elements
        # If the prediction has a same number, assign the result as '0'
        r_pred_majorityVote[np.sign(r_pred_majorityVote)==1] = 1
        r_pred_majorityVote[np.sign(r_pred_majorityVote)==-1] = -1
        r_pred_majorityVote[np.sign(r_pred_majorityVote)==0] = 0

        return r_pred_majorityVote         
    
    def fit(self, X, r):
        """
            Build a RandomeForest classifier from the training set (X, r)
            X: the feature matrix 
            r: class labels 
            
            1. Create 'num_trees' bootstrap samples
            2. Randomly select 'num_features' < d features
            3. Learn an un-pruned decision tree on each sample
            4. Determine the best split using only these features
        """

        ### Your code starts here ###
        self.X_ensamble = np.zeros((self.num_trees, np.shape(X)[0], np.shape(X)[1]))
        self.r_ensamble = np.zeros((self.num_trees, len(r)))
        self.clf_ensamble = []
        self.feature_index_list = []
        
        for dataset_index in range(self.num_trees):
            # The ensamble of the bootstrapped data
            X_bootstrap, r_bootstrap = self.bootstrapDataset(X,r)
            self.X_ensamble[dataset_index] = X_bootstrap
            self.r_ensamble[dataset_index] = r_bootstrap
            
            # The features are selected
            # Should be selected without replacement
            features = np.random.choice(np.shape(X)[1], size = self.num_features, replace=False)
            self.feature_index_list.append(features) 
                      
            # The X_bootstrap is reduced into X_bootstrap_reduced
            X_bootstrap_reduced = np.zeros((len(X_bootstrap), self.num_features))
            for index, feature in enumerate(features):
                X_bootstrap_reduced[:,index] = X_bootstrap[:,feature]
            
            # Learn a tree only with the selected features
            clf_tree = DecisionTreeClassifier(criterion = 'entropy', random_state = 0, max_depth = self.max_depth)
            clf_tree.fit(X_bootstrap_reduced, r_bootstrap)
            self.clf_ensamble.append(clf_tree)



    def predict(self,X):
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
        
        # The predictions will be made with each of the classifier
        r_pred = []
        r_pred_ensamble = np.zeros((self.num_trees, len(X)))
        X_reduced = np.zeros((len(X), self.num_features))
        
        # While classification, only the selected features for each dataset will be used
        for clf_index in range(self.num_trees):            
            clf_tree = self.clf_ensamble[clf_index]
            
            # The feature vector should be reduced with the selected features for each tree
            for index, feature in enumerate(self.feature_index_list[clf_index]):
                X_reduced[:,index] = X[:,feature]
            
            r_pred_ensamble[clf_index] = clf_tree.predict(X_reduced)
            

        for k in range(self.num_trees):
            # make predictions based on the first k tree classifiers
            # The majority vote will decide the final prediction
            r_pred_majorityVote = self.majorityVote(r_pred_ensamble[:k])
            r_pred.append(r_pred_majorityVote)
            
        return r_pred

