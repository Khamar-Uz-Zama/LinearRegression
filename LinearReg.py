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
