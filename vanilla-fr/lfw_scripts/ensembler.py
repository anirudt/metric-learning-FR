#!/usr/bin/python
from metric_learn import LMNN, ITML, LSML, SDML, NCA
from threading import Thread

mls = {
    'lmnn': LMNN(),
    'itml': ITML(),
    'lsml': LSML(),
    'sdml': SDML(),
    'nca': NCA()
    }

"""
This file accepts the split and whitened data from the 
LFW dataset and feeds it to multiple metric learning algorithms
and then feeds it further into an Ensemble classifier.

For the kNN-like probabilities, we use a modified Softmax 
for determining them.
"""

def assemble():
  """ Receives the training and test set samples and
  performs metric learning algorithms followed by an
  baseline ensemble classifier """

if __name__ == '__main__':
  assemble()
