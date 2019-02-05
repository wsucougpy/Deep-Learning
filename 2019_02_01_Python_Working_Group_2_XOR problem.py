# -*- coding: utf-8 -*-
"""
Created on Thu Jan  3 23:42:50 2019

@author: Jikhan Jeong
"""

#### 2019 Spring XOR problem with Python Working Group
#### Reference : https://github.com/gilbutITbook/006958

import numpy as np

w11 = np.array([-2,-2]) # Nand gate parameters
b1 = 3

w12 = np.array([2,2])  # Or gate parameters
b2 = -1

w2  = np.array([1,1])  # And gate parameters
b3 = -1

# Perceptron

def perceptron(x,w,b):
    y= np.sum(w*x) +b
    if y<=0:
        return 0
    else:
        return 1

# Non-and Gate (1)
        
def NAND(x1, x2):
    return perceptron(np.array([x1, x2]), w11, b1)   ## perceptron 1 (hidden layer)

# OR Gate      (2)
def OR(x1, x2):
    return perceptron(np.array([x1, x2]), w12, b2)   ## perceptron 2 (hidden layer)

# And Gate     (3)
    
def AND(x1, x2):
    return perceptron(np.array([x1, x2]), w2, b3)    ## perceptron 3 (output layer)

# XOR Gate     (3) with (1) and (2)  ---> solving XOR
    
def XOR(x1, x2):
    return AND(NAND(x1, x2), OR(x1, x2))             ## percetpron 1,2 -> perceptron 3

if __name__ == '__main__':
    for x in [(0,0),(1,0),(0,1),(1,1)]:
        y = XOR(x[0],x[1])
        print(x,y)


