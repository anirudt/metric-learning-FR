from pybrain.datasets            import SupervisedDataSet
from pybrain.utilities           import percentError
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.structure.modules   import SoftmaxLayer
import numpy as np
import pickle
from pybrain.structure           import TanhLayer
from pybrain.structure           import SigmoidLayer
import os
from sklearn                     import decomposition
import csv
from pandas                      import DataFrame
import pandas

def nn(data):
    """ Represents the NN to train our data """
    training_set = SupervisedDataSet*


    input_nodes = 3
    hidden_layer_1 = 10
    hidden_layer_2 = 10
    output_layer = 5

    net = buildNetwork(input_nodes, hidden_layer_1, hidden_layer_2, output_layer, bias=True, hiddenclass=TanhLayer)

if __name__ == '__main__':
    # Handle input of training data
    # Either read from file OR, generate random numbers like now:
    input_data_1 = np.random.rand(256)
    input_data_2 = np.random.rand(256)
    # .... Take data from more sources to feed to the NN

    input_data = np.array([input_data_1 input_data_2])

    # Call the Neural Network and pass the data

    # Handle output and compute performance metrics
