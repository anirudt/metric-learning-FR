#!/usr/bin/python
from classifier import LMNN, ITML, LSML, SDML, MLThread, RCA, NCA, LFDA, LDML
from threading import Thread
import classifier
import numpy as np
import pdb
import itertools

mls = {
    'lmnn': LMNN(),
    'itml': ITML(),
    'lsml': LSML(),
    'sdml': SDML(),
    'ldml': LDML(),
    'nca': NCA(),
    'rca': RCA(),
    'lfda': LFDA()
    }

"""
This file accepts the split and whitened data from the 
LFW dataset and feeds it to multiple metric learning algorithms
and then feeds it further into an Ensemble classifier.

For the kNN-like probabilities, we use a modified Softmax 
for determining them.
"""

# Helper for generating all subsets of a list as a set.
def list_mls(arr):
    combs = []
    for i in xrange(len(arr)+1):
        listing = [list(x) for x in itertools.combinations(arr, i)]
        combs.extend(listing)
    return combs


def generic_model_fitter_prob(ml, X_train, y_train, X_test, y_test):
  """ Takes a generic ML model and fits it with the data,
  can be used for system testing. """
  ml.fit(X_train, y_train)
  X_te = ml.transform(X_test)
  return ml.predict_proba(X_te)

def generic_model_fitter(opt, X_train, y_train, X_test, y_test):
  """ Takes a generic ML model and fits it with the data,
  can be used for unit testing. """
  ml = mls[opt]
  ml.fit(X_train, y_train)
  X_tr = ml.transform(X_train)
  X_te = ml.transform(X_test)
  accuracy, y_pred = classifier.sk_nearest_neighbour(X_tr, y_train, X_te, y_test)
  return accuracy, y_pred

# Dict for reference
str2func = {'soft': generic_model_fitter_prob,
            'hard': generic_model_fitter}

def assemble_series(X_train, y_train, X_test, y_test, weights, algos, opt):
    """ Receives the training and test set samples and
    performs metric learning algorithms followed by a
    baseline ensemble classifier, using series techniques"""

    num_classifiers = len(algos)
    num_samples = X_test.shape[0]
    num_centroids = len(np.unique(y_train))

    if opt is "soft":
        probabilities = np.zeros((num_samples, num_centroids, num_classifiers))
    else:
        all_predictions = np.zeros((num_classifiers, num_samples))
        accuracies = np.zeros((num_classifiers, 1))
        print 'Printing all accuracies'
    for idx, algo in enumerate(algos):
        # Call one by one and append pred and acc for hard
        # and proba for soft.
        if opt is "soft":
            probabilities[:,:,idx] = generic_model_fitter_prob(mls[algo], \
                    X_train, y_train, X_test, y_test)
        else:
            accuracies[idx], all_predictions[idx,:] = generic_model_fitter(algo, \
                    X_train, y_train, X_test, y_test)
            print accuracies[idx]

    # Start ensemble work.
    if opt is "soft":
        # Do necessary action
        avg = np.average(probabilities, axis=2, weights=weights)
        y_pred = np.argmax(avg, axis=1)
        c = np.sum(y_pred == y_test)
        accuracy = c*100.0/len(y_test)
        return accuracy, y_pred

    else:
        # Do necessary action
        # Across every sample, take a majority of the votes
        all_predictions = np.array(all_predictions, dtype=np.int32)
        majority_pred = np.zeros(num_samples)
        for sample in xrange(all_predictions.shape[1]):
            majority_pred[sample] = np.bincount(all_predictions[:, sample]).argmax()

        majority_pred = majority_pred.T
        majority_pred = np.array(majority_pred, dtype=np.int32)
        c = np.sum(majority_pred == y_test)
        accuracy = c * 100.0 / num_samples
        print accuracy, majority_pred
        return accuracy, majority_pred

def assemble_parallel(X_train, y_train, X_test, y_test, weights, opt):
  """ Receives the training and test set samples and
  performs metric learning algorithms followed by a
  baseline ensemble classifier using parallel techniques. """

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
    accuracies =  np.zeros(num_classifiers)
    predictions = np.zeros((num_classifiers, num_samples))
    for idx, _ in enumerate(mls):
      accuracies[idx], predictions[idx, :] = threads[idx].join()

    majority_pred = np.zeros(num_samples)
    # Run a voting classifier here on predictions.
    for sample in xrange(predictions.shape[1]):
      majority_pred[sample] = np.bincount(predictions[:, sample]).argmax()
      
      # TODO: implement a confidence score with respect to each vote count.

    majority_pred = majority_pred.T
    majority_pred = np.array(majority_pred, dtype=np.int32)
    c = np.sum(majority_pred == y_test)
    accuracy = c * 100.0 / num_samples
    return accuracy, majority_pred

if __name__ == '__main__':
  assemble()
