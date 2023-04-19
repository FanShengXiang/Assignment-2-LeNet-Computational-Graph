# -*- coding: utf-8 -*-
"""
Created on Mon Apr 17 22:55:31 2023

@author: AutoLab
"""

import numpy as np
import pandas as pd

class MLP_Two_Layers:
    """
    A 2-Layer Multi-Layer Perceptron. 
    Parameters:
    ----------
    input_size: int, default = 250
        The input size or number of features in input layer
    hidden_size: int, default = 100
        The number of nodes in the hidden layer
    output_size: int, default = 1
        The output size or the number of classes in the output layer for softmax/binary classification
    random_state: int, default = None
        Controls the randomness of the initialization of weights
    Attributes:
    ----------
    parameters: dict, default: dict containing weights and bias based on random initialization.
        Dictionary of the weights and bias after the passing through the epochs.
    velocity: dict, default: dict containing velocity matrix of the same shape as weights and bias initialized to zeros.
        Dictionary of the velocity of respective weights and bias.
    cache: dict, default=None
        Dictionary containing neuron outputs and activations.
    X: numpy.ndarray, default=None
        Training examples of input data of size (m, n_x)
    Y: numpy.ndarray
        The targets with the associated ground truths, of size (m,n_y)
    """
    def __init__(self, input_size=250, hidden_size=100, output_size=1,random_state=21):

        n_x = input_size
        n_h = hidden_size
        n_y = output_size

        np.random.seed(random_state) 
        
        W1 = np.random.randn(n_h,n_x) * 0.01  # random initialization
        b1 = np.zeros((n_h,1))
        W2 = np.random.randn(n_y,n_h) * 0.01  # random initialization
        b2 = np.zeros((n_y,1))

        self.parameters = {"W1": W1,
                           "b1": b1,
                           "W2": W2,
                           "b2": b2}

        self.velocity = {"W1": np.zeros(W1.shape),
                         "b1": np.zeros(b1.shape),
                         "W2": np.zeros(W2.shape),
                         "b2": np.zeros(b2.shape)}

        self.cache = None
        self.X = None
        self.Y = None

    def forward(self, X):
        """
        Runs a single forward propagation pass.
        Parameters:
        -----------
        X: numpy.ndarray
            Training examples of input data of size (m, n_x)
        
        Returns:
        ---------
        A2: numpy.ndarray
            The softmax/sigmoid output of the second activation
        """

        W1 = self.parameters['W1']   #(n_x,n_h)
        b1 = self.parameters['b1']   #(1,n_h)
        W2 = self.parameters['W2']   #(n_h,n_y)
        b2 = self.parameters['b2']   #(1,n_y)

        Z1 = np.dot(W1,X) + b1                                                                   #(n_h,n_x)*(n_x,m) + (n_h,1)
        A1 = np.tanh(Z1)   #apply tanh activation                                                #(n_h,m)
        Z2 = np.dot(W2,A1) + b2    
        
        if self.full_Y.shape[0]==1:                                                              #(n_y,n_h)*(n_h,m) + (n_y,1)
            A2 = 1/(1+np.exp(-Z2))   #apply sigmoid activation   #(n_y,m)
        else:
            A2 = np.exp(Z2)/np.sum(np.exp(Z2),axis=0, keepdims = True)     #apply softmax activation

        self.cache = {"Z1": Z1,
                      "A1": A1,
                      "Z2": Z2,
                      "A2": A2}
        return A2

    def loss(self, A2, Y):
        """
        Calculates the loss between outputs of forward propagation and the targets.
        Parameters:
        -----------
        A2: numpy.ndarray
            The softmax output of forward propagation
        Y: numpy.ndarray
            The targets with the associated ground truths, of size (m,n_y)
        
        Returns:
        ---------
        cost: float
            The average value of all losses between softmax output and targets.
        """

        m = Y.shape[1] # number of examples

        if Y.shape[0]==1: 
            cost = np.sum(((- np.log(A2))*Y + (-np.log(1-A2))*(1-Y)))/m  # compute cost
        else:
            loss_each_example = -np.sum(Y * np.log(A2),axis=1)
            all_losses = np.sum(loss_each_example)
            cost = all_losses/m

        return cost

    def backward(self, learning_rate=0.05, beta=0.9):
        """
        Runs a single backpropagation pass and stores the updated weights in self.parameters for the next forward propagation.
        Parameters:
        ----------
        learning_rate: float, default=0.05
            Controls the gradient descent step at each iteration.
        
        beta: float, default=0.9
            Momentum of gradient descent.
        Returns:
        -------
        None
        """
        m = self.X.shape[1] # number of examples

        # First, retrieve W1 and W2 from the dictionary "parameters".
        W1 = self.parameters['W1']
        b1 = self.parameters['b1']    
        W2 = self.parameters['W2']
        b2 = self.parameters['b2']
            
        # Retrieve also A1 and A2 from dictionary "cache".
        A1 = self.cache['A1']
        A2 = self.cache['A2']
        
        # Backward propagation: calculate dW1, db1, dW2, db2. 
        dZ2 = A2 - self.Y
        dW2 = np.dot(dZ2, A1.T) / m
        db2 = np.sum(dZ2, axis = 1, keepdims = True) / m
        dZ1 = np.multiply(np.dot(W2.T, dZ2), (1 - np.square(A1)))
        dW1 = np.dot(dZ1, self.X.T) / m
        db1 = np.sum(dZ1, axis = 1, keepdims = True) / m
        
        # Update rule for each parameter

        self.velocity['W1'] = beta*self.velocity['W1']  + (1-beta)*dW1
        self.velocity['b1'] = beta*self.velocity['b1']  + (1-beta)*db1
        self.velocity['W2'] = beta*self.velocity['W2']  + (1-beta)*dW2
        self.velocity['b2'] = beta*self.velocity['b2']  + (1-beta)*db2

        W1 = W1 - learning_rate * self.velocity['W1']
        b1 = b1 - learning_rate * self.velocity['b1']
        W2 = W2 - learning_rate * self.velocity['W2']
        b2 = b2 - learning_rate * self.velocity['b2']
        
        self.parameters = {"W1": W1,
                           "b1": b1,
                           "W2": W2,
                           "b2": b2}

    def get_batch(self, X, Y, batch_size):
        """
        Segregate the training set of X and Y into mini-batches.
        Parameters:
        ----------
        X: numpy.ndarray
            Training examples of input data of size (n_x, m)
        Y: numpy.ndarray
            The targets with the associated ground truths, of size (n_y, m)            
        batch_size: int
            Size of training batch from training set
        
        Returns:
        -------
        mini_batches: list
            List containing tuples of mini_batch_X and mini_batch_Y
        """

        m = X.shape[1]          # number of training examples

        batch_start_index = np.arange(0,m,batch_size)
        num_batches = len(batch_start_index)
        mini_batches = []

        for batch in range(1, num_batches+1):

            if batch < num_batches:
                mini_batch_X = X[:,(batch-1)*batch_size:batch*batch_size]
                mini_batch_Y = Y[:,(batch-1)*batch_size:batch*batch_size]
            elif batch == num_batches:
                mini_batch_X = X[:,(batch-1)*batch_size:]
                mini_batch_Y = Y[:,(batch-1)*batch_size:]

            mini_batch = (mini_batch_X,mini_batch_Y)
            mini_batches.append(mini_batch)

        return mini_batches 

    def fit(self, X, Y, batch_size=32, epochs=1000, learning_rate=0.05, beta=0.9):
        """
        Parameters:
        ----------
        X: numpy.ndarray or pandas.core.frame.DataFrame
            Training examples of input data of size (m, n_x)
        Y: numpy.ndarray or pandas.core.frame.DataFrame or pandas.core.series.Series
            The targets with the associated ground truths, of size (m,n_y)
        batch_size: int, default=32
            Size of training batch from training set
        epochs: int, default=1000
            Number of forward/backward passes through the entire dataset
        learning_rate: float, default=0.05
            Controls the gradient descent step at each iteration.
        beta: float, default=0.9
            Momentum of gradient descent.            
        Returns:
        -------
        parameters: dict
            Dictionary of the weights and bias after the passing through the epochs.                    
        """
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values
        if isinstance(Y, (pd.core.frame.DataFrame,pd.core.series.Series)):
            Y = Y.values

        if X.ndim > 1:
            X = X.T
        else:
            X = np.expand_dims(X, axis=1)

        if Y.ndim > 1:
            Y = Y.T
        else:
            Y = np.expand_dims(Y, axis=0)

        self.full_Y = Y
        self.full_X = X

        mini_batches = self.get_batch(X, Y, batch_size)

        for i in range(0, epochs):   

            for mini_batch in mini_batches:

                mini_batch_X, mini_batch_Y = mini_batch

                self.X = mini_batch_X
                self.Y = mini_batch_Y
                
                A2 = self.forward(mini_batch_X)         
                
                cost = self.loss(A2, mini_batch_Y)  
                
                self.backward(learning_rate,beta)

                # Print the cost every 100 epochs
            if (i+1) % 1 == 0:
                print ("Train Loss after epoch %i: %f" %(i+1, cost))

        return self.parameters

    def predict(self, X):
        """
        Parameters:
        ----------
        X: numpy.ndarray or pandas.core.frame.DataFrame
            Test examples of input data of size (m, n_x)
        Returns:
        -------            
        predictions: numpy.ndarray
            Predicted classes for each test examples
        """
        if isinstance(X, pd.core.frame.DataFrame):
            X = X.values

        if X.ndim > 1:
            X = X.T
        else:
            X = np.expand_dims(X, axis=1)

        A2 = self.forward(X)

        if A2.shape[0]==1:
            predictions = A2[0]>0.5
        else:
            predictions = (A2 == np.max(A2,axis=0)).astype(bool).T
        
        return predictions

#%%
from tqdm import tqdm
import cv2
def read_colorHist(path):
    with open(path) as f:
        label_list = []
        color_hist = np.empty((0,768))
        temp_hist = np.empty((0,768))
        for line in tqdm(f.readlines()):
            s = line.split(' ')
            print(s)
            img = cv2.imread(s[0])
            img = cv2.resize(img, (128, 128))
            # Calculate histogram without mask
            hist1 = cv2.calcHist([img],[0],None,[256],[0,256])
            hist2 = cv2.calcHist([img],[1],None,[256],[0,256])
            hist3 = cv2.calcHist([img],[2],None,[256],[0,256])
            hist = np.concatenate((hist1.T,hist2.T,hist3.T),axis=1)

            temp_hist = np.concatenate((temp_hist,hist),axis=0)

            if len(temp_hist) > 1000:
                color_hist = np.concatenate((color_hist,temp_hist),axis=0)
                temp_hist = np.empty((0,768))
                
            label_list.append(s[1][:-1])
        color_hist = np.concatenate((color_hist,temp_hist),axis=0)
        label_list = np.array(label_list)
        label_list = np.reshape(label_list, (len(label_list),1))
    return color_hist, label_list
def top1_5_score(true_y,prd_y,model):
    label = np.unique(model.full_Y)
    top1_score = 0
    top5_score = 0
    for i in range(len(true_y)):
        top5_ans = np.argpartition(prd_y[i], -5)[-5:]
        if str(int(true_y[i])) in label[top5_ans]:
            top5_score = top5_score + 1
        if str(int(true_y[i])) == label[np.argmax(prd_y[i])]:
            top1_score = top1_score + 1
    return top1_score/len(true_y) , top5_score/len(true_y)

#%%

train_x, train_y = read_colorHist("train.txt")
val_x, val_y = read_colorHist("val.txt")
test_x, test_y = read_colorHist("test.txt")
train_y = train_y.astype(np.float)
val_y = val_y.astype(np.float)
test_y = test_y.astype(np.float)
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(train_y)
train_y_o = enc.transform(train_y).toarray()
val_y_o = enc.transform(val_y).toarray()
test_y_o = enc.transform(test_y).toarray()
from sklearn.utils import shuffle
train_x_s, train_y_o_s= shuffle(train_x, train_y_o, random_state=0)
train_x_one, train_y_one= shuffle(train_x, train_y, random_state=0)
#%%
from sklearn.linear_model import Perceptron
# val_x, val_y 
# test_x, test_y
per = Perceptron(tol=1e-3, random_state=0)
per.fit(train_x_one, train_y_one)
model = MLP_Two_Layers(input_size=train_x.shape[1], hidden_size=768, output_size=50, random_state=42)
model.fit(train_x_s, train_y_o_s, batch_size=500, epochs=50, learning_rate=0.005, beta=0.9)
#%%
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import OneHotEncoder
print("One Layer Perceptron Accuracy :")
print(accuracy_score(per.predict(test_x),test_y))
print("Two Layer Perceptron Accuracy :")
print(accuracy_score(model.predict(test_x),test_y_o))
#%%