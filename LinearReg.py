# -*- coding: utf-8 -*-
"""
@author:    Khamar Uz Zama
"""

import argparse
import csv
import numpy as np

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--data', help='Data File')
parser.add_argument('--learningRate', help='Learning Rate')
parser.add_argument('--threshold', help='Threshold')

args = parser.parse_args()

filePath = args.data
lr = float(args.learningRate)
threshold = float(args.threshold)

data = []

with open(filePath) as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
       data.append(row)

data = np.array(data)

Target = data[:,-1]
Target = Target.astype(float)
Target = Target.reshape(Target.shape[0], 1)

Exes = data[:, :-1]
Exes = Exes.astype(float)
ones = np.ones(Exes.shape[0])
Exes = np.column_stack((ones,Exes))

weights = np.zeros_like(Exes[0])  
weights = weights.reshape(len(Exes[0]),1)      
weights = weights.astype(float)
no_of_weights = len(weights)

def get_preditions(features, weights):
    """
        Dot protduct features and weights (W0.X0 + W1.X1 + W2.X2) for 1000 features
        Return: 1000 predictions for 1000 features. Shape of output == (1000)
    """
   
    predictions = np.dot(features, weights)
    return predictions

def get_cost(features, targets, weights):
    """
        Calculate Squre of error.
        Shape of error = (1000, 1)
        Return: sum of all the errors
    """
  
    predictions = get_preditions(features, weights)

    error = (predictions - targets)
    sq_error = np.array(error)**2

    return sq_error.sum()

def get_weights(features, targets, weights, lr):
# ############################################################
#    DO NOT DELETE - Hardcoded for future reference - 
#    Calculating weights and gradients individually for 3 features
#    
#    x0 = features[:,0]
#    x1 = features[:,1]
#    x2 = features[:,2]
#    
#    grad_w0 = [a*b for a,b in zip(x0,T_minus_P)]
#    grad_w1 = [a*b for a,b in zip(x1,T_minus_P)]
#    grad_w2 = [a*b for a,b in zip(x2,T_minus_P)]
#    
#    grad_w0 = sum(i for i in grad_w0)
#    grad_w1 = sum(i for i in grad_w1)
#    grad_w2 = sum(i for i in grad_w2)
#   
#    weights[0][0] = weights[0][0] + (lr * (grad_w0))
#    weights[1][0] = weights[1][0] + (lr * (grad_w1))
#    weights[2][0] = weights[2][0] + (lr * (grad_w2))
#    
# ############################################################
    predictions = get_preditions(features, weights)
    T_minus_P = targets - predictions
    gradients = np.zeros_like(Exes)
    sum_of_gradients = np.zeros(Exes.shape[1])
    sum_of_gradients = sum_of_gradients.reshape(sum_of_gradients.shape[0], 1)
   
    for i in range(gradients.shape[1]):
        grad = [a*b for a,b in zip(features[:,i],T_minus_P)]
        gradients[:,i] = grad
    
    for i in range(gradients.shape[1]):
        sum_of_gradients[i] = sum(i for i in gradients[:,i])

    for i in range(gradients.shape[1]):
        weights[i][0] = weights[i][0] + (lr * (sum_of_gradients[i]))


    return weights

All_Errors = []
All_Weights = []
i = 0
All_Weights.append(weights.flatten())
previous_error = get_cost(Exes, Target, weights)
All_Errors.append([previous_error])

while(True):
    output = []

    weights = get_weights(Exes, Target, weights, lr)
    current_error = get_cost(Exes, Target, weights)
    All_Errors.append([current_error])
    x = weights.flatten()
    All_Weights.append(x)

    if ((previous_error - current_error) < threshold):
        break
    if(i%10 == 0):
        print(i, "--", previous_error - current_error)    
    previous_error = current_error
    i += 1
    
with open('output.csv', 'w', newline='') as myfile:
    """
        Output the results to csv file
    """
    writer = csv.writer(myfile, quoting=csv.QUOTE_ALL)
    for i in range(len(All_Weights)):
         output = []
         output.append(i)
         for j in range(no_of_weights):
             output.append(round(All_Weights[i][j], 4))
         output.append(round(All_Errors[i][0], 4))
         writer.writerow(output)
