# -*- coding: utf-8 -*-
"""
Created on Thu Jan 31 21:36:51 2019

@author: Jikhan Jeong
"""

import os
print("Current Working Directory " , os.getcwd())
os.chdir("C:/python/a_python/2019_working_group/")


import numpy as np
import pandas as pd


## Basic Structure

data = pd.read_csv("2019_02_01_python_working_group.csv", header=None, index_col=False, names=['y1','x1','x2','x3','x4','x5','x6','x7','x8','x9','x10','x11','x12'])
data.head(3) # 3 row x 13 col
len(data) # 156 row


X = data.loc[:,"x1":"x12"] # training features
Y = data.loc[:,"y1"] # training label
X
Y[1]
X.shape[1]    # 12
len(X)        # 155
type(X)       # dataframe
type(Y)       # series
X.shape[1]    # 12
len(X.loc[1]) #12
np.sign(Y[1]*np.dot(np.zeros(X.shape[1]),X.loc[1])) 


def perceptron(feature, label, iters):
    
    w = np.zeros(feature.shape[1]) # array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
    final_iter = iters
    
    for iter in range(iters): # For Each Training Iteration to Max
      
        error = 0
        yes = 0      

        for i in range(len(feature)): ## all sample n = 155
            
            y = label[i]         # actual value in i row (1)
            x = feature.loc[i]     # feature vector in i row
            a = np.dot(w, x) # Predicted value (2) 
            m = np.sign(y*a) # (Margins) m>0 correct, m<0 wrong
            if m <= 0:       # (wrong guess) if the prediciton is wrong 
                w = w + np.dot(x, y)  # update the weight, b is not considered 
                error += 1   # update the numer of error             
            else:            # (correct guess)
                yes += 1     
                pass                
        print("iter: {}".format(iter), "Accuracy: {}".format(1-error/len(feature)))   
    return  w, error, yes, final_iter,1-error/len(X)

perceptron(X,Y,20)
