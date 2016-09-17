#!/usr/bin/python
import numpy as np
from time import time
import cv2
import pdb
import csv
import classifier
import dataPorter
from metric_learn import LMNN, ITML_Supervised, LSML_Supervised, SDML_Supervised
from sklearn.cross_validation import train_test_split

""" Script for EigenFace Method of Face Recognition """

"""
TODO:
Some helper functions need to be written for better analysis.
1. An automated function for displaying the set of images in a subplot, followed by the graphify function.
2. Compute the eigenvectors and analyse them -- compare, visualize and save them.
"""


NUM_IMGS         = 40
IMGS_PER_PERSON  = 4
NUM_PEOPLE       = NUM_IMGS / IMGS_PER_PERSON
TOT_IMGS_PP = 10

"""
Acceptable tuples include:
KGP_DB = (20, 2, (100, 100))
ATT_DB = (40, 4, (92, 112))
"""

# Following are string to XXX conversion dicts.
str2classifier = {"pca": classifier.PCA(),
                  "lda": classifier.LDA(),
                  "lbp": classifier.LBP(),
                  "lmnn": classifier.LMNN()}

str2traindatabase = { "KGP": dataPorter.import_custom_training_set,
                      "ATT": dataPorter.import_att_training_set}

str2testdatabase = { "KGP": dataPorter.import_custom_testing_set,
                     "ATT": dataPorter.import_att_testing_set}

str2complete = { "KGP": None,
                 "ATT": dataPorter.import_att_complete_set,
                 "LFW": dataPorter.import_lfw_complete_set}

str2ravelling = {"pca": 'ravel',
                 "pcalda": 'ravel',
                 "lbp": 'unravel',
                 "lmnn": 'ravel'}

mls = [
        LMNN,
        ITML_Supervised,
        SDML_Supervised,
        LSML_Supervised
        ]

"""
Some helper functions.
"""
def display_imgs(face_matrix):
    """
    TODO: Reshape, quantise and display the images in a subplot fashion.
    http://stackoverflow.com/questions/17111525/how-to-show-multiple-images-in-one-figure
    """

def eigen_logger(eigen_vals, eigen_vecs):
    """
    TODO: Identify the class of photos and place the eigenvalues and eigenvectors in a fairly understandable
    data structure. Also, try to plot the eigenvalue projections across all data fitted hitherto.
    Things to save: del A, del (eig_vals A), del (eig_vecs A)
    """
    #pdb.set_trace()
    with open("eigen_vals.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(eigen_vals.tolist())
    with open("eigen_vecs.csv", "wb") as f:
        writer = csv.writer(f)
        writer.writerows(eigen_vecs.tolist())

""" Hybrid Classifiers Definition """
def pca_lda(face_matrix, pca, lda, labels):
    print "shape ", face_matrix.shape
    selected_eigen_vecs_pca, eigen_face_space = pca.fit(face_matrix, labels)

    # TODO: Return something
    selected_eigen_vecs_lda, lda_projection = lda.fit(eigen_face_space, labels)

    return [2, pca, lda, selected_eigen_vecs_pca, selected_eigen_vecs_lda, lda_projection]

def train(classifier_str, database, face_matrix, labels):
    """ Get data, train, get the Eigenvalues and store them."""
    print face_matrix.shape

    if classifier_str in str2classifier:
        model = str2classifier[classifier_str]
        trained_bundle = model.fit(face_matrix, labels)
        return [1, model, trained_bundle]
        # TODO: Handle this trained_bundle in a standard way
    else:
        if classifier_str == "pcalda":
            pca = str2classifier[classifier_str[0:3]]
            lda = str2classifier[classifier_str[3:]]
            trained_bundle = pca_lda(face_matrix, pca, lda, labels)
            return trained_bundle

def test(classifier_str, database, X_test, y_test, y_train, trained_bundle):
    """ Acquire a new image and get the data. """
    if trained_bundle[0] == 1:
        model = trained_bundle[1]
        test_proj, space = model.transform(X_test)
        accuracy = classifier.sk_nearest_neighbour(space.T, y_train, test_proj.T, y_test)
        return accuracy

        # FIXME: Remove this once the LBP has an option to give its space key
        if hasattr(space, 'shape'):
            detected_idx = classifier.nearest_neighbour(space, test_proj)
            print detected_idx
            detected_idx = int((detected_idx/IMGS_PER_PERSON))+1
            print "Detected argmin: ", detected_idx
        else:
            detected_idx = test_proj
            print space
        print detected_idx
        #pdb.set_trace()

    elif trained_bundle[0] == 2:
        model1 = trained_bundle[1]
        test_proj1, _ = model1.transform(test_image)

        model2 = trained_bundle[2]
        test_proj2, space = model2.transform(test_proj1)

        detected_idx = classifier.nearest_neighbour(space, test_proj2)
        detected_idx = int((detected_idx)/IMGS_PER_PERSON)+1
        print detected_idx

    # Log the accuracy metrics.
    return [person, classifier_str, detected_idx]
    
    #print "Detected face is of serial no. {0}".format((detected_idx+2)/IMGS_PER_PERSON)

def multi_runner(classifier_str, database):
    """
    Runs the training and test for all the different tilted faces. Returns a list of lists of eigenvalues and eigenvectors.
    """
    g = open("results.csv", 'ab')
    numfolds = 4
    a = open("accuracy_"+classifier_str+"_"+str(numfolds)+".csv", "wb")
    wr = csv.writer(g)
    acc = csv.writer(a)
    eigenvals, eigenvecs = [], []
    training_set_idx = range(1,TOT_IMGS_PP+1)
    num = 0
    for person in range(1,NUM_PEOPLE+1):
        c = 0
        for fold in range(TOT_IMGS_PP):
            testing_this_round = training_set_idx[fold:fold+1]
            training_this_round = (training_set_idx[:fold] + training_set_idx[fold+1:])[:numfolds]
            trained_bundle = train(classifier_str, database, training_this_round)
            print testing_this_round, training_this_round
            # TODO: Compute a cross validation score here
            # to prevent overfitting.
            # Source: http://stackoverflow.com/questions/16379313/how-to-use-the-a-k-fold-cross-validation-in-scikit-with-naive-bayes-classifier-a
            for tilt_idx in testing_this_round:
                val = test(person, tilt_idx, trained_bundle, classifier_str, database)
                print val[2], val[0]
                wr.writerow(val)
                if val[2] == val[0]:
                    c+=1
        accuracy = c*100.0/TOT_IMGS_PP
        num += accuracy
        tmp = [classifier_str, database, person, accuracy]
        acc.writerow(tmp)
    num /= NUM_PEOPLE
    acc.writerow([num])


    return 0

def new_multi_runner(classifier_str, database):
    X, y = str2complete[database](str2ravelling[classifier_str])
    numfolds = 4
    a = open("accuracy_"+classifier_str+"_"+str(numfolds)+".csv", "wb")
    acc = csv.writer(a)
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=1.0/numfolds,random_state=42)
    # Train the model now.
    trained_bundle = train(classifier_str, database, X_train, y_train)

    # val should be your accuracy.
    accuracy = test(classifier_str, database, X_test, y_test, y_train, trained_bundle)

    # To store accuracies of several classifiers.
    return accuracy

def main(classifier_str, database):
    global NUM_PEOPLE, NUM_IMGS, IMGS_PER_PERSON, dims
    if database == "KGP":
        (NUM_PEOPLE, NUM_IMGS, IMGS_PER_PERSON, TOT_IMGS_PP, dims) = (10, 20, 2, 10, (100, 100))
    elif database == "ATT":
        (NUM_PEOPLE, NUM_IMGS, IMGS_PER_PERSON, TOT_IMGS_PP, dims) = (10, 40, 4, 10, (92, 112))
    print "sizes are ",(NUM_IMGS, IMGS_PER_PERSON, dims) 

    #multi_runner(classifier_str, database)
    accuracy = new_multi_runner(classifier_str, database)
    return accuracy

if __name__ == '__main__':
    acc1 = main("pca", "ATT")
    print "Accuracy for PCA: ", acc1
    main("pcalda", "ATT")
