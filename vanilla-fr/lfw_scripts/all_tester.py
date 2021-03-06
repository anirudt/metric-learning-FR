"""
===================================================
Faces recognition example using eigenfaces and SVMs
===================================================

The dataset used in this example is a preprocessed excerpt of the
"Labeled Faces in the Wild", aka LFW_:

  http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz (233MB)

.. _LFW: http://vis-www.cs.umass.edu/lfw/

Expected results for the top 5 most represented people in the dataset:

================== ============ ======= ========== =======
                   precision    recall  f1-score   support
================== ============ ======= ========== =======
     Ariel Sharon       0.67      0.92      0.77        13
     Colin Powell       0.75      0.78      0.76        60
  Donald Rumsfeld       0.78      0.67      0.72        27
    George W Bush       0.86      0.86      0.86       146
Gerhard Schroeder       0.76      0.76      0.76        25
      Hugo Chavez       0.67      0.67      0.67        15
       Tony Blair       0.81      0.69      0.75        36

      avg / total       0.80      0.80      0.80       322
================== ============ ======= ========== =======

"""
from __future__ import print_function

from time import time
import logging
import matplotlib.pyplot as plt

import pdb
import classifier
from sklearn.lda import LDA
from sklearn.cross_validation import train_test_split
from sklearn.datasets import fetch_lfw_people
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import RandomizedPCA
from sklearn.svm import SVC
from ensembler import generic_model_fitter, assemble_parallel, assemble_series, list_mls, cleanCachedMls
import numpy as np

# Helper Function
def getStr(arr):
    return ' '.join(arr)


def main(opt_list, arg_list, runall=False):
    """ Pass either only_ml, ml_svm, or only_svm"""
    accuracies = {
            'soft_unw': [],
            'soft_wei': [],
            'hard_wei': [],
            'hard_unw': [],
            'svm': 0,
            'lda': 0
            }
    #print(__doc__)

    # Display progress logs on stdout
    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(message)s')


    ###############################################################################
    # Download the data, if not already on disk and load it as numpy arrays

    lfw_people = fetch_lfw_people(min_faces_per_person=70, resize=0.4)

    # introspect the images arrays to find the shapes (for plotting)
    n_samples, h, w = lfw_people.images.shape

    # for machine learning we use the 2 data directly (as relative pixel
    # positions info is ignored by this model)
    X = lfw_people.data
    n_features = X.shape[1]

    # the label to predict is the id of the person
    y = lfw_people.target
    target_names = lfw_people.target_names
    n_classes = target_names.shape[0]

    print("Total dataset size:")
    print("n_samples: %d" % n_samples)
    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)


    ###############################################################################
    # Split into a training set and a test set using a stratified k fold

    # split into a training and testing set
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42)


    ###############################################################################
    # Compute a PCA (eigenfaces) on the face dataset (treated as unlabeled
    # dataset): unsupervised feature extraction / dimensionality reduction
    n_components = 150

    print("Extracting the top %d eigenfaces from %d faces"
          % (n_components, X_train.shape[0]))
    t0 = time()
    pca = RandomizedPCA(n_components=n_components, whiten=True).fit(X_train)
    print("done in %0.3fs" % (time() - t0))

    eigenfaces = pca.components_.reshape((n_components, h, w))
    print("Projecting the input data on the eigenfaces orthonormal basis")
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    print("done in %0.3fs" % (time() - t0))

    if opt_list is "serial":
        if not runall:
            a = time()
            acc, y_pred = assemble_series(X_train_pca, y_train, X_test_pca, y_test, ['lmnn', 'lsml', 'rca', 'ldml', 'lfda'], 'soft', "weighted")
            print("accuracy = %s",acc)
            print(classification_report(y_test, y_pred, target_names=target_names))
            print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
            b = time()

        else:
            if 'soft_unw' in arg_list:
                mls = list_mls(['lmnn', 'lsml', 'rca', 'lfda', 'ldml'])
                ml_strs = []
                y_preds = []
                for ml in mls:
                    if len(ml) == 0:
                        continue
                    print(ml)
                    acc, y_pred = assemble_series(X_train_pca, y_train, X_test_pca, y_test, ml, 'soft', 'unweighted')
                    y_preds.append(y_pred)
                    accuracies['soft_unw'].append(acc)
                    ml_strs.append(getStr(ml))
                    print("accuracy = %s",acc)
                    print(classification_report(y_test, y_pred, target_names=target_names))
                    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
                y_preds = np.array(y_preds)
                num_samples = y_preds.shape[1]
                majority_pred = np.zeros(num_samples)
                
                for sample in xrange(y_preds.shape[1]):
                    majority_pred[sample] = np.bincount(y_preds[:,sample]).argmax()
                majority_pred= np.array(majority_pred, dtype=np.int32)
                c = np.sum(majority_pred == y_test)
                accuracy = c * 100.0 / num_samples
                accuracies['soft_unw'].append(accuracy)
                ml_strs.append('all')
                cleanCachedMls()
            if 'soft_wei' in arg_list:
                mls = list_mls(['lmnn', 'lsml', 'rca', 'lfda', 'ldml'])
                ml_strs = []
                y_preds = []
                for ml in mls:
                    if len(ml) == 0:
                        continue
                    print(ml)
                    acc, y_pred = assemble_series(X_train_pca, y_train, X_test_pca, y_test, ml, 'soft', 'weighted')
                    y_preds.append(y_pred)
                    accuracies['soft_wei'].append(acc)
                    ml_strs.append(getStr(ml))
                    print("accuracy = %s",acc)
                    print(classification_report(y_test, y_pred, target_names=target_names))
                    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
                y_preds = np.array(y_preds)
                num_samples = y_preds.shape[1]
                majority_pred = np.zeros(num_samples)
                
                for sample in xrange(y_preds.shape[1]):
                    majority_pred[sample] = np.bincount(y_preds[:,sample]).argmax()
                majority_pred= np.array(majority_pred, dtype=np.int32)
                c = np.sum(majority_pred == y_test)
                accuracy = c * 100.0 / num_samples
                accuracies['soft_wei'].append(accuracy)
                ml_strs.append('all')

                cleanCachedMls()
            if 'hard_wei' in arg_list:
                mls = list_mls(['lmnn', 'lsml', 'rca', 'lfda', 'ldml'])
                ml_strs = []
                y_preds = []
                for ml in mls:
                    if len(ml) == 0:
                        continue
                    print(ml)
                    acc, y_pred = assemble_series(X_train_pca, y_train, X_test_pca, y_test, ml, 'hard', 'weighted')
                    y_preds.append(y_pred)
                    accuracies['hard_wei'].append(acc)
                    ml_strs.append(getStr(ml))
                    print("accuracy = %s",acc)
                    print(classification_report(y_test, y_pred, target_names=target_names))
                    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
                y_preds = np.array(y_preds, dtype=np.int32)
                num_samples = y_preds.shape[1]
                majority_pred = np.zeros(num_samples)
                
                for sample in xrange(y_preds.shape[1]):
                    majority_pred[sample] = np.bincount(y_preds[:,sample]).argmax()
                majority_pred= np.array(majority_pred, dtype=np.int32)
                c = np.sum(majority_pred == y_test)
                accuracy = c * 100.0 / num_samples
                accuracies['hard_wei'].append(accuracy)
                ml_strs.append('all')
                cleanCachedMls()
            if 'hard_unw' in arg_list:
                mls = list_mls(['lmnn', 'lsml', 'rca', 'lfda', 'ldml'])
                ml_strs = []
                y_preds = []
                for ml in mls:
                    if len(ml) == 0:
                        continue
                    print(ml)
                    acc, y_pred = assemble_series(X_train_pca, y_train, X_test_pca, y_test, ml, 'hard', 'unweighted')
                    y_preds.append(y_pred)
                    accuracies['hard_unw'].append(acc)
                    ml_strs.append(getStr(ml))
                    print("accuracy = %s",acc)
                    print(classification_report(y_test, y_pred, target_names=target_names))
                    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
                y_preds = np.array(y_preds, dtype=np.int32)
                num_samples = y_preds.shape[1]
                majority_pred = np.zeros(num_samples)
                
                for sample in xrange(y_preds.shape[1]):
                    majority_pred[sample] = np.bincount(y_preds[:,sample]).argmax()
                majority_pred= np.array(majority_pred, dtype=np.int32)
                c = np.sum(majority_pred == y_test)
                accuracy = c * 100.0 / num_samples
                accuracies['hard_unw'].append(accuracy)
                ml_strs.append('all')
                cleanCachedMls()
    if opt_list is "parallel":
        """ TODO:  Opt for the parallel thread implementation. """
        if not runall:
            a = time()
            acc, y_pred = assemble_parallel(X_train_pca, y_train, X_test_pca, y_test, 'hard')
            print("accuracy = %s",acc)
            print(classification_report(y_test, y_pred, target_names=target_names))
            print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
            b = time()
            print("Total time taken for all this: {0}".format(b-a))

        else:
            mls = list_mls(['lmnn', 'lsml', 'rca', 'lfda', 'ldml'])
            ml_strs = []
            y_preds = []
            for ml in mls:
                if len(ml) == 0:
                    continue
                print(ml)
                acc, y_pred = assemble_parallel(X_train_pca, y_train, X_test_pca, y_test, 'hard')
                accuracies['hard'].append(acc)
                y_preds.append(y_pred)
                ml_strs.append(getStr(ml))
                print("accuracy = %s", acc)
                print(classification_report(y_test, y_pred, target_names=target_names))
                print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

            y_preds = np.array(y_preds)
            num_samples = y_preds.shape[1]
            majority_pred = np.zeros(num_samples)
            
            for sample in xrange(y_preds.shape[1]):
                majority_pred[sample] = np.bincount(y_preds[:,sample]).argmax()
            majority_pred= np.array(majority_pred, dtype=np.int32)
            c = np.sum(majority_pred == y_test)
            accuracy = c * 100.0 / num_samples
            accuracies['hard'].append(accuracy)
            ml_strs.append('all')


    ###############################################################################
    print("Without the LMNN structure")
    # Train a SVM classification model

    print("Fitting the classifier to the training set")
    t0 = time()
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf'), param_grid)
    clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")
    print(clf.best_estimator_)

    ###############################################################################
    # Quantitative evaluation of the model quality on the test set

    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    acc = 100.0*sum(y_pred == y_test) / len(y_test)
    print("accuracy = %s",acc)
    print("done in %0.3fs" % (time() - t0))

    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

    print("Fitting the classifier to the training set")
    t0 = time()
    clf = LDA()
    clf.fit(X_train_pca, y_train)
    print("done in %0.3fs" % (time() - t0))
    print("Best estimator found by grid search:")

    ###############################################################################
    # Quantitative evaluation of the model quality on the test set

    print("Predicting people's names on the test set")
    t0 = time()
    y_pred = clf.predict(X_test_pca)
    acc1 = 100.0*sum(y_pred == y_test) / len(y_test)
    print("accuracy = %s",acc1)
    print("done in %0.3fs" % (time() - t0))

    print(classification_report(y_test, y_pred, target_names=target_names))
    print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))

    if runall:
        accuracies['svm'] = acc
        accuracies['lda'] = acc1
        ml_strs.append('svm')
        ml_strs.append('lda')
        ml_strs = ", ".join(ml_strs)
        return ml_strs, accuracies


def run_many_epochs(num_epochs):
    accuracies = {
            'soft_unw': [],
            'soft_wei': [],
            'hard_wei': [],
            'hard_unw': [],
            'svm': [],
            'lda': []
            }
    for i in xrange(num_epochs):
        headers, acc = main('serial', ['hard_wei', 'soft_wei', 'hard_unw', 'soft_unw'], runall=True)
        if len(acc['soft_unw']) > 0:
            accuracies['soft_unw'].append(acc['soft_unw'])
        if len(acc['soft_wei']) > 0:
            accuracies['soft_wei'].append(acc['soft_wei'])
        if len(acc['hard_wei']) > 0:
            accuracies['hard_wei'].append(acc['hard_wei'])
        if len(acc['hard_unw']) > 0:
            accuracies['hard_unw'].append(acc['hard_unw'])
        accuracies['svm'].append(acc['svm'])
        accuracies['lda'].append(acc['lda'])
        cleanCachedMls()

    # Write all results.
    for key in accuracies.keys():
        arr = accuracies[key]
        np.savetxt('logs/results_'+str(num_epochs)+'_'+key+'serial.csv', arr, delimiter=',', header=headers)


if __name__ == "__main__":
    #main("series", runall=True)
    run_many_epochs(24)
