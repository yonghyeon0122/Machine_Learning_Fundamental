import numpy as np

import torch
import torch.nn as nn


# Fully connected neural network with one hidden layer
class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_epochs, learning_rate=0.1):

        '''
        input_size: [int], feature dimension 
        hidden_size: number of hidden nodes in the hidden layer
        output_size: number of classes in the dataset, 
        max_epochs: maximum number of epochs to run stochastic gradient descent
        learning_rate: learning rate for SGD
        '''
        ### Your Code starts here
        ### You want to construct your MLP Here (consider the recommmended functions in HW4 writeup)       
        
        
        
        
        super(MLP, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.max_epoch = max_epochs
        self.learning_rate = learning_rate
        
        self.Linear_1 = nn.Linear(input_size, hidden_size)
        self.Linear_2 = nn.Linear(hidden_size, output_size)
        self.ReLU = nn.ReLU()
        
        

    def forward(self, x):
        ### To do feed-forward pass
        ### Your Code starts here
        ### Use the layers you constructed in __init__ and pass x through the network
        hidden = self.ReLU(self.Linear_1(x))
        out = self.Linear_2(hidden)        
        
        return out

    def fit(self,dataloader,criterion,optimizer):

        '''
        Function used to training the MLP

        Inputs:
        dataloader: includes the feature matrix and classlabels corresponding to the training set 
        criterion: the loss function used. Set to cross_entropy loss!
        optimizer: which optimization method to train the model. Use SGD here! 

        Returns:
        Training loss: cross-entropy loss evaluated on the training dataset
        Training Accuracy: Prediction accuracy (0-1 loss) evaluated on the training dataset
        '''

        train_loss = []
        train_acc = []
        
        for i in range(self.max_epoch):

            for j,(images,labels) in enumerate(dataloader):
                
                # Forward pass (consider the recommmended functions in HW4 writeup)
                out = self.forward(images.reshape(-1, 28*28)) 
                out_max, out_pred = torch.max(out, 1)
                loss = criterion(out, labels) # loss calculation 
                # Backward pass and optimize (consider the recommmended functions in HW4 writeup)
                                             
                optimizer.zero_grad() # set the gradients of all optimized parameters to zero
                loss.backward() # loss backward propagation               
                optimizer.step() # parameter update
                # Track the accuracy
                pred_acc = torch.zeros(len(out))
                for idx in range(len(out)):
                    if out_pred[idx] == labels[idx]:
                        pred_acc[idx] = 1
                    else:
                        pred_acc[idx] = 0
                
            train_loss.append(loss.item()) # loss append
            train_acc.append(torch.mean(pred_acc).item()) # accuracy append
            
        train_loss = np.asarray(train_loss)
        train_acc = np.asarray(train_acc)
            
        return train_loss,train_acc


    def predict(self,dataloader,criterion):
        '''
        Function used to evaluate the MLP

        Inputs:
        dataloader: includes the feature matrix and classlabels corresponding to the validation/test set 
        criterion: the loss function used. Set to cross_entropy loss!

        Returns:
        Test loss: cross-entropy loss evaluated on the validation/test set 
        Test Accuracy: Prediction accuracy (0-1 loss) evaluated on the validation/test set 
        '''
        
        with torch.no_grad():
            for j,(images,labels) in enumerate(dataloader):

                # compute output and loss
                out = self.forward(images.reshape(-1, 28*28))
                out_max, out_pred = torch.max(out, 1)
                loss = criterion(out, labels) # loss calculation
                # measure accuracy and record loss
                test_loss = loss
                pred_acc = torch.zeros(len(out))
                for idx in range(len(out)):
                    if out_pred[idx] == labels[idx]:
                        pred_acc[idx] = 1
                    else:
                        pred_acc[idx] = 0
                test_acc = torch.mean(pred_acc)
        
        return test_loss, test_acc

