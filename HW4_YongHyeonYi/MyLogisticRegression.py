import numpy as np



class MyLogisticRegression:
    """
    
    2-class logistic regression with one set of parameters (W, w0)
    label: 0,1
    posterior probability is given    
    
    Convergence of gradient descent can be determined by checking the error difference between current and previous iteration
    Error change threshold: 1e-6
    """
        
    
    def __init__(self,d,max_iter=1000,eta=1e-3):

        '''
        d: [int], feature dimension used to initialize weights
        max_iter: maximum number of iterations to run gradient descent
        eta: learning rate for gradient descent
        '''
        ### Your Code starts here
        self.d = d
        self.max_iter = max_iter
        self.eta = eta
        self.W = np.random.normal(0, 0.1, d) # The initial weight is a normal distribution, and already includes w0
        # self.W = np.zeros(d)
        self.E_diff_threshold = 1e-6
        # Sigmoid clip boundary must be decided
        # The sigmoid values must be clipped between [epsilon, 1-epsilon]
        # Thus the log(exponent) should be clipped between [log(1-epsilon)-log(epsilon), log(epsilon)-log(1-epsilon)]
        epsilon = 1e-10
        self.sigmoid_boundary = [epsilon, 1-epsilon]
        
        self.sigmoid_logExponent_boundary = [np.log(epsilon) - np.log(1-epsilon), np.log(1-epsilon)-np.log(epsilon),  ]
    
    def sigmoid(self, W, X):
        # Returns posterior probability
        # The sigmoid values must be clipped between [epsilon, 1-epsilon]
        # Thus the log(exponent) should be clipped between [log(1-epsilon)-log(epsilon), log(epsilon)-log(1-epsilon)]
        log_exponent = np.matmul(np.transpose(W), X)
        
        if log_exponent < self.sigmoid_logExponent_boundary[0]:
            sigmoid = self.sigmoid_boundary[0] # sigmoid lower boundary
        elif log_exponent > self.sigmoid_logExponent_boundary[1]:
            sigmoid = self.sigmoid_boundary[1] # sigmoid upper boundary
        else:
            exponent = np.exp(np.matmul(np.transpose(W), X))
            sigmoid = exponent / (1 + exponent) # normal sigmoid calculation
               
        
        return sigmoid # A very small value is added to avoid numerical problems

    def error(self, W, X, r):
        # Calculate error value for each iteration
        n_sample = len(X)
        E=0
        y = np.zeros(n_sample)
        
        for t in range(n_sample):
            y[t] = self.sigmoid(W, X[t])
            E += ( r[t] * np.log(y[t]) + (1 - r[t]) * np.log(1-y[t]) )
        E = -E # negative log likelihood
        
        return E
    
    def weight_gradient(self, W, X, r, eta):
        # Calculate the error gradient for the gradient descent
        n_sample = len(X)
        
        y = np.zeros(n_sample)
        W_gradient = np.zeros(len(W))
                
        for j in range(len(W)):
            for t in range(n_sample):
                y[t] = self.sigmoid(W, X[t])
                W_gradient[j] += eta * (r[t] - y[t]) * X[t][j]
              
        
        return W_gradient

    def preProcess(self, X):
        # Z-scoring each features
        X_new = np.zeros(X.shape)
        X_mean = np.mean(X[:,0:-1], axis=1) # mean of each sample
        X_std = np.std(X[:,0:-1], axis=1) # std of each sample
                
        for t, Xt in enumerate(X):
            X_new[t] = np.subtract(X[t], X_mean[t]) / X_std[t]
            
        X_new[:,-1] = 1 # the last column is all one 
        
        return X_new

    def fit(self, X, r):

        ### Your Code starts here
        ### Don't forget to check convergence!
        E_current = 0
        E_prev = 0
        n_sample = len(X)
        
        # self.eta = self.eta/ n_sample  # Step size depends on the number of samples
        
        f = open("Convergence.txt", "a")
        f.write("==== Logistic Regression ====\n")  
        f.close()
        
        # Z-scoring features
        X = self.preProcess(X)
        
        for iteration in range(self.max_iter):        
           
            
            E = self.error(self.W, X, r)
            W_gradient = self.weight_gradient(self.W, X, r, self.eta)
            self.W = self.W + W_gradient
                     
            print("Iteration: " + str(iteration) + " Error: " + str(E) + "\n")
 
            # Calculate the error difference
            E_current = E
            if(iteration > 0):
                E_difference = E_prev - E_current
                if(E_difference > 0 and E_difference < self.E_diff_threshold):
                    break # convergence
            E_prev = E_current

        f = open("Convergence.txt", "a")
        f.write("==== E final: " + str(E) + " Iteration: " + str(iteration) + " ====\n")        
        f.close()
        
        return 0

    def predict(self,X):
        ### Your Code starts here
        # Calculate the discriminant and make predictions
        n_sample = len(X)
        g = np.zeros(n_sample)
        P_C1 = np.zeros(n_sample)
        r_pred = np.zeros(n_sample)
              
        # Z-scoring features
        X = self.preProcess(X)
        
        
        for t in range(n_sample):      
            P_C1[t] = self.sigmoid(self.W,  X[t])
            if P_C1[t] > 0.5:
                r_pred[t] = 1
            else:
                r_pred[t] = 0
            
            
        return r_pred

    
