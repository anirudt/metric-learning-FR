#!/usr/bin/python
from classifier import LMNN, ITML, LSML, SDML, MLThread, RCA, NCA, LFDA, LDML
from threading import Thread
import classifier
import numpy as np
import pdb
import itertools

mls = {
    'lmnn': LMNN(),
    #'itml': ITML(),
    'lsml': LSML(),
    #'sdml': SDML(),
    'ldml': LDML(),
    #'nca': NCA(),
    'rca': RCA(),
    'lfda': LFDA()
    }

learned_hard_ypreds = {
        'lmnn': None,
        'lsml': None,
        'ldml': None,
        'rca': None,
        'lfda': None
        }

learned_hard_acc = {
        'lmnn': None,
        'lsml': None,
        'ldml': None,
        'rca': None,
        'lfda': None
        }

learned_hard_wts = {
        'lmnn': None,
        'lsml': None,
        'ldml': None,
        'rca': None,
        'lfda': None
        }

learned_soft_prob = {
        'lmnn': None,
        'lsml': None,
        'ldml': None,
        'lfda': None,
        'rca': None
        }

learned_soft_wts = {
        'lmnn': None,
        'lsml': None,
        'ldml': None,
        'lfda': None,
        'rca': None
        }

"""
This file accepts the split and whitened data from the 
LFW dataset and feeds it to multiple metric learning algorithms
and then feeds it further into an Ensemble classifier.

For the kNN-like probabilities, we use a modified Softmax 
for determining them.
"""

def cleanCachedMls():
  # Iterate tthrough all keys of learned_mls and clear cache
  for key in learned_soft_prob.keys():
      learned_soft_prob[key] = None
      learned_soft_wts[key] = None
      learned_hard_ypreds[key] = None
      learned_hard_acc[key] = None
      learned_hard_wts[key] = None


# Helper for generating all subsets of a list as a set.
def list_mls(arr):
    combs = []
    for i in xrange(len(arr)+1):
        listing = [list(x) for x in itertools.combinations(arr, i)]
        combs.extend(listing)
    return combs


def generic_model_fitter_prob(ml_str, X_train, y_train, X_test, y_test, algo_opts=None):
  """ Takes a generic ML model and fits it with the data,
  can be used for system testing."""
  if learned_soft_prob[ml_str] is not None:
      if algo_opts is "weighted":
          return learned_soft_prob[ml_str], learned_soft_wts[ml_str]
      else:
          return learned_soft_prob[ml_str]
  ml = mls[ml_str]
  ml.fit(X_train, y_train)
  X_tr = ml.transform(X_train)
  X_te = ml.transform(X_test)
  probabilities = ml.predict_proba(X_te)
  if algo_opts is "weighted":
      # TODO: 
      accuracy, y_pred = classifier.sk_nearest_neighbour(X_tr, y_train, X_tr, y_train)
      training_error = (100.0 - accuracy) / 100.0
      weight = np.log((1 - training_error)/training_error)
      learned_soft_prob[ml_str] = probabilities
      learned_soft_wts[ml_str] = weight
      return probabilities, weight
  else:
      learned_soft_prob[ml_str] = probabilities
      return probabilities

def generic_model_fitter(ml_str, X_train, y_train, X_test, y_test, algo_opts=None):
  """ Takes a generic ML model and fits it with the data,
  can be used for unit testing."""
  if learned_hard_ypreds[ml_str] is not None:
      if algo_opts is "weighted":
          return learned_hard_acc[ml_str], learned_hard_ypreds[ml_str], learned_hard_wts[ml_str]
      else:
          return learned_hard_acc[ml_str], learned_hard_ypreds[ml_str]

  ml = mls[ml_str]
  ml.fit(X_train, y_train)
  X_tr = ml.transform(X_train)

  if algo_opts is "weighted":
    # Measure training accuracy
    accuracy, y_pred = classifier.sk_nearest_neighbour(X_tr, y_train, X_tr, y_train)
    training_error = (100.0 - accuracy) / 100.0
    weight = np.log((1 - training_error)/training_error)
    X_te = ml.transform(X_test)
    accuracy, y_pred = classifier.sk_nearest_neighbour(X_tr, y_train, X_te, y_test)
    learned_hard_acc[ml_str], learned_hard_ypreds[ml_str], learned_hard_wts[ml_str] = accuracy, y_pred, weight
    return accuracy, y_pred, weight
  else:
    X_te = ml.transform(X_test)
    accuracy, y_pred = classifier.sk_nearest_neighbour(X_tr, y_train, X_te, y_test)
    learned_hard_ypreds[ml_str], learned_hard_acc[ml_str] = y_pred, accuracy
    return accuracy, y_pred

# Dict for reference
str2func = {'soft': generic_model_fitter_prob,
            'hard': generic_model_fitter}

def assemble_series(X_train, y_train, X_test, y_test, algos, opt, algo_opts=None):
    """ Receives the training and test set samples and
    performs metric learning algorithms followed by a
    baseline ensemble classifier, using series techniques"""

    num_classifiers = len(algos)
    num_samples = X_test.shape[0]
    num_centroids = len(np.unique(y_train))
    weights = np.zeros(num_classifiers)

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
            if algo_opts is "weighted":
                print algo, algos
                #pdb.set_trace()
                probabilities[:,:,idx],  weights[idx] = generic_model_fitter_prob(algo, \
                    X_train, y_train, X_test, y_test, "weighted")
            else:
                probabilities[:,:,idx] = generic_model_fitter_prob(algo, \
                    X_train, y_train, X_test, y_test)

            
        else:
            if algo_opts is "weighted":
                print algo, algos
                #pdb.set_trace()
                accuracies[idx], all_predictions[idx,:], weights[idx] = generic_model_fitter(algo, \
                    X_train, y_train, X_test, y_test, "weighted")
            else:
                accuracies[idx], all_predictions[idx,:] = generic_model_fitter(algo, \
                    X_train, y_train, X_test, y_test)
            print accuracies[idx]

    if algo_opts is "weighted":
        wts = np.array(weights)
        wts = wts / np.linalg.norm(wts)
    else:
        wts = np.ones(num_classifiers)
    # Start ensemble work.
    if opt is "soft":
        # Do necessary action
        print "I was here"
        print "Weights are ", wts
        avg = np.average(probabilities, axis=2, weights=wts)
        y_pred = np.argmax(avg, axis=1)
        c = np.sum(y_pred == y_test)
        accuracy = c*100.0/len(y_test)
        return accuracy, y_pred

    else:
        # Do necessary action
        # Across every sample, take a majority of the votes
        # For now, we are not considering weights for Hard Voting, however,
        # in the future, we can use a similar probability matrix for implementing the same.
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

def assemble_parallel(X_train, y_train, X_test, y_test, opt, algo_opts=None):
  """ Receives the training and test set samples and
  performs metric learning algorithms followed by a
  baseline ensemble classifier using parallel techniques. """

  threads = [None]*len(mls.keys())

  # Step 1: Spawn threads and generate class probabilities
  num_classifiers = len(mls.keys())
  num_samples = X_test.shape[0]
  num_centroids = len(np.unique(y_train))

  for idx, key in enumerate(mls):
    threads[idx] = MLThread(target=str2func[opt],args=(key, X_train, y_train, X_test, y_test, algo_opts))
    threads[idx].start()

  # Step 2: Combine all of them and use a baseline ensemble classifier
  if opt is "soft":
    probabilities = np.zeros((num_samples, num_centroids, num_classifiers))
    for idx, _ in enumerate(mls):
      tmp = threads[idx].join() 
      if algo_opts is "weighted":
          weights[idx] = tmp[1]
          probabilities[:,:,idx] = tmp[0]
      else:
          probabilities[:,:,idx] = tmp
      

    
    if not algo_opts:
         weights = np.ones(num_classifiers)
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

    predictions = np.array(predictions, dtype=np.int32)
    majority_pred = np.zeros(num_samples)
    # Run a voting classifier here on predictions.
    for sample in xrange(predictions.shape[1]):
      majority_pred[sample] = np.bincount(predictions[:, sample]).argmax()
      
    majority_pred = majority_pred.T
    majority_pred = np.array(majority_pred, dtype=np.int32)
    c = np.sum(majority_pred == y_test)
    accuracy = c * 100.0 / num_samples
    return accuracy, majority_pred

if __name__ == '__main__':
  assemble()
