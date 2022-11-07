import numpy as np



class MySVM:
    """
    2- class SVM
    label r: [-1, 1]
    lambda_val: regularization parameter
    """
    def __init__(self,d,lambda_val=1,max_iter=1000,eta=1e-3):

        '''
        d: [int], feature dimension used to initialize weights
        lambda_val: the regularization parameter \lambda
        max_iter: maximum number of iterations to run gradient descent
        eta: learning rate for gradient descent
        '''
        ### Your Code starts here
        self.d = d
        self.lambda_val = lambda_val
        self.max_iter = max_iter
        self.eta = eta
        self.W = np.zeros(d) # The initial weight is a normal distribution, and already includes w0
        self.E_diff_threshold = 1e-6

    def error(self, W, X, r):
        # Calculate error value for each iteration
        n_sample = len(X)
        E=0
        
        for t in range(n_sample):
            # E += (1/n_sample) * ( np.max([0, 1 - r[t] * np.matmul(np.transpose(W), X[t])]) + self.lambda_val/2 * np.matmul(np.transpose(W), W))  
            hinge_loss = np.amax([0, 1 - r[t] * np.matmul(np.transpose(W), X[t])])
            E += 1/n_sample *( hinge_loss + self.lambda_val/2 * np.matmul(np.transpose(W), W) )
            
        return E

    def preProcess(self, X):
        # Z-scoring each features
        X_new = np.zeros(X.shape)
        X_mean = np.mean(X[:,0:-1], axis=1) # mean of each sample
        X_std = np.std(X[:,0:-1], axis=1) # std of each sample
                
        for t, Xt in enumerate(X):
            X_new[t] = np.subtract(X[t], X_mean[t]) / X_std[t]
            
        X_new[:,-1] = 1 # the last column is all one 
        
        return X_new

    def weight_gradient(self, W, X, r, eta):
        
        n_sample = len(X)
        W_gradient = np.zeros(len(W))
        prediction = np.zeros(n_sample)
        
        X = self.preProcess(X)
        
        for t in range(n_sample):
            prediction[t] = r[t] * np.matmul(np.transpose(W), X[t])
            if prediction[t] > 1:
                W_gradient += -1 * eta * self.lambda_val * self.W
            elif prediction[t] <= 1:
                W_gradient += eta * (r[t] * X[t] - self.lambda_val * self.W)
        
        return W_gradient

    def fit(self,X,r):

        ### Your Code starts here
        ### Don't forget to check convergence!
        E_current = 0
        E_prev = 0
        n_sample = len(X)
               
        
        f = open("Convergence.txt", "a")
        f.write("==== SVM ====\n")  
        f.close()

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
        r_pred = np.zeros(n_sample)

        X = self.preProcess(X)
        
        for t in range(n_sample):
            g[t] = np.matmul(np.transpose(self.W), X[t])
        r_pred = np.sign(g)   

        return r_pred
