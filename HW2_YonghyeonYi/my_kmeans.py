import numpy as np

np.random.seed(0)



def kmeans(X,k=5):
    '''The kmeans performs the Kmeans algorithm
    # The parameters received are:
    # - X (N x 3): Matrix for a color image RGB, where N is the number of pixels. 
    # - k (1 x 1): Number of clusters (number of colors in the compression)
    # 
    # The function should return:
    # - r (N x K): Contains 0 or 1, where r(n,k) contains 1 if pixel n belongs to cluster k, otherwise 0
    # - mu (k x 3): Contains the k centroids found, representing the k colors learned
    # - E : reconstruction errors during training
    '''



    max_iter = 100
    '''initial mu'''
    N = len(X) # number of samples
    m_idx = np.random.permutation(N)
    mu = X[m_idx[:k],:] # (k x 3) <-- mu starts with random rows in X
    E_now=0 
    E = np.array([])

    # Iteration starts
    for i in range(max_iter):

        # update []r
        ### YOUR CODE STARTS HERE ###
        r = np.zeros((N,k))
        distance = np.zeros((N, k))
        # Record the Euclidean distance information        
        for cluster_index in range(k):
            for sample_index, x in enumerate(X):
                distance[sample_index][cluster_index] = np.linalg.norm(x - mu[cluster_index])
        
        # Compare the distance between each cluster's mu
        # Assign '1' only for the maximum cluster
        for sample_index in range(N):
            min_cluster = np.argmin(distance[sample_index], axis=0)
            r[sample_index][min_cluster] = 1
        

        # Calculate the total reconstruction error define in Textbook Eq.(7.3) 
        ### YOUR CODE STARTS HERE ###
        
        # update b
        # if the Euclidean distance is minimum among the clusters, assign '1'
        b = np.zeros((N,k))
        E_prev = E_now
        E_now = 0
        for sample_index in range(N):
            min_cluster = np.argmin(distance[sample_index], axis=0)
            b[sample_index][min_cluster] = 1
        
        # calculate the reconstruction error, E
        for sample_index in range(N):
            for cluster_index in range(k):
                E_now += b[sample_index][cluster_index] * distance[sample_index][cluster_index]
        
        E = np.append(E, [E_now])
        
        print('Iteration {}: Error {}'.format(i,E_now))

        

        # update mu
        ### YOUR CODE STARTS HERE ###
        
        # calculate mu using r and the feature vector
        normalizer = np.sum(r, axis=0)
        mu = np.zeros((k, len(X[0])))
        
        for cluster_index in range(k):
            for sample_index, x in enumerate(X):
                mu_temp = r[sample_index][cluster_index] * x.reshape(1,-1)
                mu[cluster_index] += mu_temp.reshape(-1)
            if(normalizer[cluster_index] != 0.0):
                mu[cluster_index] = mu[cluster_index] / normalizer[cluster_index]
        
        # print(normalizer)   
        # print(mu)
        # check convergence
        # by checking if the the error function decreased less than 1e-6 from the previous iteration.
        # Break loop if converged
        ### YOUR CODE STARTS HERE ###
        error_tolerance = 1e-6
        if(i >= 1):
            if(E_prev - E_now >= 0 and E_prev - E_now < error_tolerance):
                break


    return r,mu,E
        

