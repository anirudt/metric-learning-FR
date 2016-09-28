#!/usr/bin/python
from classifier import LMNN, ITML, LSML, SDML
from metric_learn import NCA
from threading import Thread
import classifier

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

def generic_model_fitter_prob(ml, X_train, y_train, X_test):
  """ Takes a generic ML model and fits it with the data,
  can be used for system testing. """
  ml.fit(X_train, y_train)
  X_te = ml.transform(X_test)
  return ml.predict_proba(X_te)

def generic_model_fitter(opt, X_train, y_train, X_test, y_test):
  """ Takes a generic ML model and fits it with the data,
  can be used for unit testing. """
  ml = mls[opt]
  X_tr = ml.fit(X_train, y_train).transform(X_train)
  X_te = ml.transform(X_test)
  accuracy, y_pred = classifier.sk_nearest_neighbour(X_tr, y_train, X_te, y_test)
  return accuracy, y_pred

def assemble(X_train, y_train, X_test, y_test):
  """ Receives the training and test set samples and
  performs metric learning algorithms followed by a
  baseline ensemble classifier """

  threads = [None]*len(mls.keys())
  # Step 1: Spawn threads and generate class probabilities
  num_classifiers = len(mls.keys())
  num_samples = X_test.shape[0]
  num_centroids = len(np.unique(y_train))

  probabilities = np.zeros((num_samples, num_centroids, num_classifiers))
  for idx, ml in enumerate(mls):
    thread[idx] = MLThread(target=generic_model_fitter_prob,args=(ml, X_train, y_train, X_test))
    thread[idx].start()

  for idx, _ in enumerate(mls):
    probabilities[:,:,idx] = thread[idx].join()
  # Step 2: Combine all of them and use a baseline ensemble classifier

  # Get the classifier

if __name__ == '__main__':
  assemble()
