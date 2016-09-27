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

def generic_model_fitter(ml, X_train, y_train, X_test, y_test):
    """ Takes a generic ML model and fits it with the data """
    ml.fit(X_train, y_train)


def assemble(X_train, y_train, X_test, y_test):
  """ Receives the training and test set samples and
  performs metric learning algorithms followed by a
  baseline ensemble classifier """

  # Step 1: Spawn threads and generate class probabilities

  # Step 2: Combine all of them and use a baseline ensemble classifier

  # Get the classifier
  


if __name__ == '__main__':
  assemble()
