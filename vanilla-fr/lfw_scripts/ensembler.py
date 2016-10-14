#!/usr/bin/python
from classifier import LMNN, ITML, LSML, SDML, MLThread
from metric_learn import NCA
from threading import Thread
import classifier
import numpy as np

mls = {
    'lmnn': LMNN(),
    #'itml': ITML(),
    'lsml': LSML(),
    #'sdml': SDML(),
    'nca': NCA(),
    'rca': RCA()
    }

"""
This file accepts the split and whitened data from the 
LFW dataset and feeds it to multiple metric learning algorithms
and then feeds it further into an Ensemble classifier.

For the kNN-like probabilities, we use a modified Softmax 
for determining them.
"""

str2func = {'soft': generic_model_fitter_prob,
            'hard': generic_model_fitter}

def generic_model_fitter_prob(ml, X_train, y_train, X_test, y_test):
  """ Takes a generic ML model and fits it with the data,
  can be used for system testing. """
  ml.fit(X_train, y_train)
  X_te = ml.transform(X_test)
  return ml.predict_proba(X_te)

def generic_model_fitter(ml, X_train, y_train, X_test, y_test):
  """ Takes a generic ML model and fits it with the data,
  can be used for unit testing. """
  ml = mls[opt]
  X_tr = ml.fit(X_train, y_train).transform(X_train)
  X_te = ml.transform(X_test)
  accuracy, y_pred = classifier.sk_nearest_neighbour(X_tr, y_train, X_te, y_test)
  return accuracy, y_pred

def assemble(X_train, y_train, X_test, y_test, weights, opt):
  """ Receives the training and test set samples and
  performs metric learning algorithms followed by a
  baseline ensemble classifier """

  threads = [None]*len(mls.keys())

  # Step 1: Spawn threads and generate class probabilities
  num_classifiers = len(mls.keys())
  num_samples = X_test.shape[0]
  num_centroids = len(np.unique(y_train))

  for idx, key in enumerate(mls):
    ml = mls[key]
    threads[idx] = MLThread(target=str2func[opt],args=(ml, X_train, y_train, X_test, y_test))
    threads[idx].start()

  # Step 2: Combine all of them and use a baseline ensemble classifier
  if opt is "soft":
    probabilities = np.zeros((num_samples, num_centroids, num_classifiers))
    for idx, _ in enumerate(mls):
      probabilities[:,:,idx] = threads[idx].join()

    avg = np.average(probabilities, axis=2, weights=weights)

    # Get the classifier
    y_pred = np.argmax(avg, axis=1)
    c = np.sum(y_pred == y_test)
    accuracy = c * 100.0 / len(y_test)
    return accuracy, y_pred
  
  else:
    accuracies =  np.zeros((num_classifiers, 1))
    predictions = np.zeros((num_classifiers, num_samples))
    for idx, _ in enumerate(mls):
      accuracies[idx], predictions[idx, :] = threads[idx].join()

    majority_pred = np.zeros((num_samples, 1))
    # Run a voting classifier here on predictions.
    for sample in xrange(predictions.shape[1]):
      majority_pred[sample] = np.bincount(predictions[:, sample]).argmax()
      
      # TODO: implement a confidence score with respect to each vote count.

    c = np.sum(majority_pred == y_test)
    accuracy = c * 100.0 / num_samples
    return accuracy, majority_pred

if __name__ == '__main__':
  assemble()
