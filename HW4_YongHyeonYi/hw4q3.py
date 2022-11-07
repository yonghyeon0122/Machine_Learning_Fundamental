import numpy as np

import torch
import torch.nn as nn

import torchvision
from torchvision.transforms import Compose, ToTensor, Normalize
from MyMLP import MLP

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# for plot
import matplotlib.pyplot as plt
plt.switch_backend('agg')

# for correctly download the dataset using torchvision, do not change!
from six.moves import urllib
opener = urllib.request.build_opener()
opener.addheaders = [('User-agent', 'Mozilla/5.0')]
urllib.request.install_opener(opener)

def convert_data_to_numpy(dataset):
    X = []
    r = []
    for i in range(len(dataset)):
       X.append(dataset[i][0][0].flatten().numpy())# flatten it to 1d vector
       r.append(dataset[i][1])

    X = np.array(X)
    r = np.array(r)

    return X,r

transforms = Compose([ToTensor(), Normalize((0.1307,), (0.3081,))])

train_dataset = torchvision.datasets.MNIST(root='MNIST-data',
                                           train=True,
                                           download=True,
                                           transform=transforms)

test_dataset = torchvision.datasets.MNIST(root='MNIST-data', 
                                          train=False,
                                          transform=transforms)


batch_size = 128

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=False)


test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


input_size = 28*28
hidden_size = 128
output_size = 10

learning_rate = 0.1

max_epochs = 10

model = MLP(input_size, hidden_size, output_size, max_epochs=max_epochs,learning_rate=learning_rate)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

train_loss,train_acc = model.fit(train_loader,criterion,optimizer)

test_loss,test_acc = model.predict(test_loader,criterion)

fig = plt.plot(np.arange(max_epochs),train_loss)
plt.xlabel('No. of Epoch')
plt.ylabel('Training Loss')
plt.savefig('hw4q3.png')

# convert pytorch dataset to numpy format for use with sklearn logistic regression and SVM - DO NOT CHANGE
train_X,train_r = convert_data_to_numpy(train_dataset)
test_X,test_r = convert_data_to_numpy(test_dataset)

# ### Your code for sklearn logistic regression and SVM starts here
clf1 = LogisticRegression(penalty='l2', solver='lbfgs', max_iter=500, multi_class='multinomial')
clf1.fit(train_X, train_r)
predict1 = clf1.predict(test_X)
score1 = clf1.score(test_X, test_r) # ratios of the correct prediction

clf2 = SVC(gamma='scale', C=10, max_iter=500)
clf2.fit(train_X, train_r)
predict2 = clf2.predict(test_X)
score2 = clf2.score(test_X, test_r) # ratios of the correct prediction

f = open("Accuracy_comparison.txt", "w")
f.write( "==== MySVM training accuracy: " + str(test_acc.item()) +   " ====\n")
f.write( "==== LogisticRegression training accuracy: " + str(score1) + " ====\n")
f.write( "==== SVC training accuracy: " + str(score2) +   " ====\n")

f.close()

